#include "Configuration.hpp"

#include <fmt/format.h>
#include <panda/Logger.h>
#include <panda/utils/Assert.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cuda/refinement/RefinementParameters.cuh>
#include <cuda/simulation/Simulation.cuh>
#include <exception>
#include <fstream>
#include <glm/ext/vector_float3.hpp>
#include <limits>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <string>
#include <string_view>

#include "utils/Serializers.hpp"  // NOLINT(misc-include-cleaner)

using json = nlohmann::json;

namespace sph::utils
{
namespace
{
struct DomainAxis
{
    float min;
    float max;
    std::string_view name;
};

void requireFinite(float value, std::string_view name)
{
    panda::expect(std::isfinite(value), fmt::format("{} must be finite", name));
}

void requirePositive(float value, std::string_view name)
{
    requireFinite(value, name);
    panda::expect(value > 0.F, fmt::format("{} must be greater than zero", name));
}

void requireNonNegative(float value, std::string_view name)
{
    requireFinite(value, name);
    panda::expect(value >= 0.F, fmt::format("{} must not be negative", name));
}

auto getGridSpacing(float min, float max, uint32_t count) -> float
{
    if (count == 1)
    {
        return 0.F;
    }
    return (max - min - (2.F * initialParticleMargin)) / static_cast<float>(count - 1);
}

void warnAboutGridCompatibility(const InitialParameters& initial,
                                const cuda::Simulation::Parameters& simulation,
                                const glm::vec3& spacing)
{
    const auto particleDiameter = 2.F * simulation.baseParticleRadius;
    const auto hasMismatch =
        (initial.particleCount.x > 1 && std::abs(spacing.x - particleDiameter) > 0.1F * particleDiameter) ||
        (initial.particleCount.y > 1 && std::abs(spacing.y - particleDiameter) > 0.1F * particleDiameter) ||
        (initial.particleCount.z > 1 && std::abs(spacing.z - particleDiameter) > 0.1F * particleDiameter);
    if (hasMismatch)
    {
        panda::log::Warning("Initial fluid grid pitch differs from boundary spacing by more than 10%");
    }
    const auto largestPitch = std::max({spacing.x, spacing.y, spacing.z});
    if (largestPitch > 0.F && (2.F * simulation.baseSmoothingRadius) < largestPitch)
    {
        panda::log::Warning("Wendland support radius is smaller than the largest initial particle pitch");
    }
}

void validateSimulationParameters(const cuda::Simulation::Parameters& simulation)
{
    const auto axes = std::array {
        DomainAxis {.min = simulation.domain.min.x, .max = simulation.domain.max.x, .name = "x"},
        DomainAxis {.min = simulation.domain.min.y, .max = simulation.domain.max.y, .name = "y"},
        DomainAxis {.min = simulation.domain.min.z, .max = simulation.domain.max.z, .name = "z"}
    };
    for (const auto& axis : axes)
    {
        requireFinite(axis.min, "simulationParameters.domain.min");
        requireFinite(axis.max, "simulationParameters.domain.max");
        const auto extent = axis.max - axis.min;
        panda::expect(
            std::isfinite(extent) && extent > (2.F * initialParticleMargin),
            fmt::format("Domain extent on {} must be finite and exceed twice the initial particle margin", axis.name));
    }
    requireNonNegative(simulation.domain.friction, "simulationParameters.domain.friction");
    requireFinite(simulation.gravity.x, "simulationParameters.gravity");
    requireFinite(simulation.gravity.y, "simulationParameters.gravity");
    requireFinite(simulation.gravity.z, "simulationParameters.gravity");
    requirePositive(simulation.speedOfSound, "simulationParameters.speedOfSound");
    requirePositive(simulation.restDensity, "simulationParameters.restDensity");
    requireNonNegative(simulation.viscosityConstant, "simulationParameters.viscosityConstant");
    requireNonNegative(simulation.surfaceTensionConstant, "simulationParameters.surfaceTensionConstant");
    requirePositive(simulation.maxVelocity, "simulationParameters.maxVelocity");
    requirePositive(simulation.baseSmoothingRadius, "simulationParameters.baseSmoothingRadius");
    requirePositive(simulation.baseParticleRadius, "simulationParameters.baseParticleRadius");
    requirePositive(simulation.baseParticleMass, "simulationParameters.baseParticleMass");
    panda::expect(simulation.threadsPerBlock >= 1 && simulation.threadsPerBlock <= 1024,
                  "simulationParameters.threadsPerBlock must be in the range [1, 1024]");
}

void validateInitialParticleDistribution(const InitialParameters& initial,
                                         const cuda::Simulation::Parameters& simulation)
{
    const auto spacing =
        glm::vec3 {getGridSpacing(simulation.domain.min.x, simulation.domain.max.x, initial.particleCount.x),
                   getGridSpacing(simulation.domain.min.y, simulation.domain.max.y, initial.particleCount.y),
                   getGridSpacing(simulation.domain.min.z, simulation.domain.max.z, initial.particleCount.z)};
    warnAboutGridCompatibility(initial, simulation, spacing);
    if (initial.particleCount.x == 1 || initial.particleCount.y == 1 || initial.particleCount.z == 1)
    {
        return;
    }
    const auto expectedMass = simulation.restDensity * spacing.x * spacing.y * spacing.z;
    panda::expect(std::isfinite(expectedMass) && expectedMass > 0.F,
                  "Initial spacing and rest density produce an invalid particle mass");
    const auto relativeMassErrorPercent = 100.F * ((simulation.baseParticleMass - expectedMass) / expectedMass);
    if (std::abs(relativeMassErrorPercent) > 10.F)
    {
        panda::log::Warning(
            "Initial particle mass mismatch: configured {:.6g}, expected {:.6g} "
            "(restDensity {:.6g}, grid spacing {:.6g} x {:.6g} x {:.6g}); "
            "deviation {:+.1f}%. Check baseParticleMass, restDensity, domain, and particleCount.",
            simulation.baseParticleMass,
            expectedMass,
            simulation.restDensity,
            spacing.x,
            spacing.y,
            spacing.z,
            relativeMassErrorPercent);
    }
}

void validateRefinementParameters(const std::optional<cuda::refinement::RefinementParameters>& parameters,
                                  uint32_t initialParticleCount)
{
    if (!parameters.has_value() || !parameters->enabled)
    {
        return;
    }
    const auto& refinement = *parameters;
    panda::expect(refinement.maxParticleCount >= initialParticleCount,
                  "refinementParameters.maxParticleCount must be at least the initial particle count");
    panda::expect(refinement.maxParticleCount <= std::numeric_limits<uint32_t>::max() / 2,
                  "refinementParameters.maxParticleCount exceeds the supported capacity");
    requirePositive(refinement.minMassRatio, "refinementParameters.minMassRatio");
    requirePositive(refinement.maxMassRatio, "refinementParameters.maxMassRatio");
    panda::expect(refinement.maxMassRatio > refinement.minMassRatio,
                  "refinementParameters.maxMassRatio must exceed minMassRatio");
    requireFinite(refinement.maxBatchRatio, "refinementParameters.maxBatchRatio");
    panda::expect(refinement.maxBatchRatio > 0.F && refinement.maxBatchRatio <= 1.F,
                  "refinementParameters.maxBatchRatio must be in the range (0, 1]");
    panda::expect(refinement.cooldown > 0, "refinementParameters.cooldown must be greater than zero");

    const auto& splitting = refinement.splitting;
    requirePositive(splitting.epsilon, "refinementParameters.splitting.epsilon");
    requirePositive(splitting.alpha, "refinementParameters.splitting.alpha");
    requirePositive(splitting.centerMassRatio, "refinementParameters.splitting.centerMassRatio");
    requirePositive(splitting.vertexMassRatio, "refinementParameters.splitting.vertexMassRatio");
    const auto childMassRatio =
        static_cast<float>(cuda::refinement::constants::splitChildCount) * splitting.vertexMassRatio;
    const auto totalSplitMassRatio = splitting.centerMassRatio + childMassRatio;
    panda::expect(std::abs(totalSplitMassRatio - 1.F) <= 1e-6F,
                  "refinementParameters.splitting mass ratios must sum to 1 across the center and 12 children");
}

void validatePoiseuilleFlowParameters(const cuda::ScenarioParameters& scenario)
{
    if (!scenario.poiseuilleFlow.has_value())
    {
        panda::expect(false, "Poiseuille flow requires scenarioParameters.poiseuilleFlow");
        return;
    }
    requireFinite((*scenario.poiseuilleFlow).drivingForce, "scenarioParameters.poiseuilleFlow.drivingForce");
}

void validateLidDrivenCavityParameters(const cuda::ScenarioParameters& scenario)
{
    if (!scenario.lidDrivenCavity.has_value())
    {
        panda::expect(false, "Lid-driven cavity requires scenarioParameters.lidDrivenCavity");
        return;
    }
    requireFinite((*scenario.lidDrivenCavity).lidVelocity, "scenarioParameters.lidDrivenCavity.lidVelocity");
}

void validateDamBreakParameters(const cuda::ScenarioParameters& scenario)
{
    if (!scenario.damBreak.has_value())
    {
        panda::expect(false, "Dam break requires scenarioParameters.damBreak");
        return;
    }
    const auto& damBreak = *scenario.damBreak;
    for (int axis = 0; axis < 3; ++axis)
    {
        const auto& domain = damBreak.expandedDomain;
        requireFinite(domain.min[axis], "scenarioParameters.damBreak.expandedDomain.min");
        requireFinite(domain.max[axis], "scenarioParameters.damBreak.expandedDomain.max");
        const auto extent = domain.max[axis] - domain.min[axis];
        panda::expect(std::isfinite(extent) && extent > (2.F * initialParticleMargin),
                      "Dam-break domain extent must be finite and exceed twice the initial particle margin");
    }
    requireNonNegative(damBreak.expandedDomain.friction, "scenarioParameters.damBreak.expandedDomain.friction");
}

void validateScenarioParameters(const std::optional<cuda::ScenarioParameters>& parameters)
{
    if (!parameters.has_value())
    {
        return;
    }
    const auto& scenario = *parameters;
    requireNonNegative(scenario.cooldown, "scenarioParameters.cooldown");
    switch (scenario.type)
    {
    case cuda::ScenarioType::Unspecified:
        break;
    case cuda::ScenarioType::PoiseuilleFlow:
        validatePoiseuilleFlowParameters(scenario);
        break;
    case cuda::ScenarioType::LidDrivenCavity:
        validateLidDrivenCavityParameters(scenario);
        break;
    case cuda::ScenarioType::DamBreak:
        validateDamBreakParameters(scenario);
        break;
    }
}
}

auto loadConfigurationFromFile(const std::string& filePath) -> std::optional<Configuration>
{
    try
    {
        std::ifstream file(filePath);
        if (!file.is_open())
        {
            panda::log::Error("Failed to open configuration file: {}", filePath);
            return {};
        }

        json jsonFile;
        file >> jsonFile;

        return jsonFile.get<Configuration>();
    }
    catch (const json::exception& e)
    {
        panda::log::Error("JSON parsing error: {}", e.what());
        return {};
    }
    catch (const std::exception& e)
    {
        panda::log::Error("Error loading configuration: {}", e.what());
        return {};
    }
}

auto validateConfiguration(const Configuration& configuration) -> void
{
    const auto initial = panda::expect(configuration.initialParameters, "Configuration is missing initialParameters");
    const auto simulation =
        panda::expect(configuration.simulationParameters, "Configuration is missing simulationParameters");
    const auto particleCount = initial.getScalarCount();
    validateSimulationParameters(simulation);
    validateInitialParticleDistribution(initial, simulation);
    validateRefinementParameters(configuration.refinementParameters, particleCount);
    validateScenarioParameters(configuration.scenarioParameters);
}

}
