#include "ConfigurationManager.hpp"

#include <panda/Logger.h>

#include <cmath>
#include <cstdint>
#include <cuda/refinement/RefinementParameters.cuh>
#include <exception>
#include <fstream>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_uint3.hpp>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <string>

#include "cuda/Simulation.cuh"

using json = nlohmann::json;

namespace sph
{

auto ConfigurationManager::loadFromFile(const std::string& filePath) -> bool
{
    try
    {
        std::ifstream file(filePath);
        if (!file.is_open())
        {
            panda::log::Error("Failed to open configuration file: {}", filePath);
            return false;
        }

        json jsonFile;
        file >> jsonFile;

        if (jsonFile.contains("simulation"))
        {
            parseSimulationParameters(jsonFile["simulation"]);
        }

        if (jsonFile.contains("refinement"))
        {
            parseRefinementParameters(jsonFile["refinement"]);
        }

        if (jsonFile.contains("initial"))
        {
            parseInitialParameters(jsonFile["initial"]);
        }
        if (jsonFile.contains("benchmark"))
        {
            parseBenchmarkParameters(jsonFile["benchmark"]);
        }

        return true;
    }
    catch (const json::exception& e)
    {
        panda::log::Error("JSON parsing error: {}", e.what());
        return false;
    }
    catch (const std::exception& e)
    {
        panda::log::Error("Error loading configuration: {}", e.what());
        return false;
    }
}

auto ConfigurationManager::loadFromString(const std::string& jsonString) -> bool
{
    try
    {
        const auto jsonFile = json::parse(jsonString);

        if (jsonFile.contains("simulation"))
        {
            parseSimulationParameters(jsonFile["simulation"]);
        }

        if (jsonFile.contains("refinement"))
        {
            parseRefinementParameters(jsonFile["refinement"]);
        }

        if (jsonFile.contains("initial"))
        {
            parseInitialParameters(jsonFile["initial"]);
        }
        if (jsonFile.contains("benchmark"))
        {
            parseBenchmarkParameters(jsonFile["benchmark"]);
        }

        return true;
    }
    catch (const json::exception& e)
    {
        panda::log::Error("JSON parsing error: {}", e.what());
        return false;
    }
    catch (const std::exception& e)
    {
        panda::log::Error("Error loading configuration: {}", e.what());
        return false;
    }
}

//NOLINTNEXTLINE(readability-function-cognitive-complexity)
void ConfigurationManager::parseSimulationParameters(const json& jsonFile)
{
    cuda::Simulation::Parameters params {};

    if (jsonFile.contains("domain"))
    {
        const auto& domain = jsonFile["domain"];
        if (domain.contains("min") && domain["min"].is_array() && domain["min"].size() == 3)
        {
            params.domain.min =
                glm::vec3(domain["min"][0].get<float>(), domain["min"][1].get<float>(), domain["min"][2].get<float>());
        }

        if (domain.contains("max") && domain["max"].is_array() && domain["max"].size() == 3)
        {
            params.domain.max =
                glm::vec3(domain["max"][0].get<float>(), domain["max"][1].get<float>(), domain["max"][2].get<float>());
        }
    }

    if (jsonFile.contains("gravity") && jsonFile["gravity"].is_array() && jsonFile["gravity"].size() == 3)
    {
        params.gravity = glm::vec3(jsonFile["gravity"][0].get<float>(),
                                   jsonFile["gravity"][1].get<float>(),
                                   jsonFile["gravity"][2].get<float>());
    }

    if (jsonFile.contains("restDensity"))
    {
        params.restDensity = jsonFile["restDensity"].get<float>();
    }

    if (jsonFile.contains("pressureConstant"))
    {
        params.pressureConstant = jsonFile["pressureConstant"].get<float>();
    }

    if (jsonFile.contains("nearPressureConstant"))
    {
        params.nearPressureConstant = jsonFile["nearPressureConstant"].get<float>();
    }

    if (jsonFile.contains("restitution"))
    {
        params.restitution = jsonFile["restitution"].get<float>();
    }

    if (jsonFile.contains("baseSmoothingRadius"))
    {
        params.baseSmoothingRadius = jsonFile["baseSmoothingRadius"].get<float>();
    }

    if (jsonFile.contains("viscosityConstant"))
    {
        params.viscosityConstant = jsonFile["viscosityConstant"].get<float>();
    }

    if (jsonFile.contains("maxVelocity"))
    {
        params.maxVelocity = jsonFile["maxVelocity"].get<float>();
    }

    if (jsonFile.contains("baseParticleRadius"))
    {
        params.baseParticleRadius = jsonFile["baseParticleRadius"].get<float>();
    }

    if (jsonFile.contains("baseParticleMass"))
    {
        params.baseParticleMass = jsonFile["baseParticleMass"].get<float>();
    }

    if (jsonFile.contains("threadsPerBlock"))
    {
        params.threadsPerBlock = jsonFile["threadsPerBlock"].get<uint32_t>();
    }

    _simulationParams = params;
}

//NOLINTNEXTLINE(readability-function-cognitive-complexity)
void ConfigurationManager::parseRefinementParameters(const json& jsonFile)
{
    cuda::refinement::RefinementParameters params;

    if (jsonFile.contains("enabled"))
    {
        params.enabled = jsonFile["enabled"].get<bool>();
    }

    if (jsonFile.contains("criterionType"))
    {
        params.criterionType = jsonFile["criterionType"].get<std::string>();
    }

    if (jsonFile.contains("minMassRatio"))
    {
        params.minMassRatio = jsonFile["minMassRatio"].get<float>();
    }

    if (jsonFile.contains("maxMassRatio"))
    {
        params.maxMassRatio = jsonFile["maxMassRatio"].get<float>();
    }

    if (jsonFile.contains("maxParticleCount"))
    {
        params.maxParticleCount = jsonFile["maxParticleCount"].get<uint32_t>();
    }

    if (jsonFile.contains("maxBatchRatio"))
    {
        params.maxBatchRatio = jsonFile["maxBatchRatio"].get<float>();
    }

    if (jsonFile.contains("initialCooldown"))
    {
        params.initialCooldown = jsonFile["initialCooldown"].get<uint32_t>();
    }

    if (jsonFile.contains("cooldown"))
    {
        params.cooldown = jsonFile["cooldown"].get<uint32_t>();
    }

    if (jsonFile.contains("splitting"))
    {
        const auto& splitting = jsonFile["splitting"];

        if (splitting.contains("epsilon"))
        {
            params.splitting.epsilon = splitting["epsilon"].get<float>();
        }

        if (splitting.contains("alpha"))
        {
            params.splitting.alpha = splitting["alpha"].get<float>();
        }

        if (splitting.contains("centerMassRatio"))
        {
            params.splitting.centerMassRatio = splitting["centerMassRatio"].get<float>();
        }

        if (splitting.contains("vertexMassRatio"))
        {
            params.splitting.vertexMassRatio = splitting["vertexMassRatio"].get<float>();
        }
    }

    if (jsonFile.contains("velocity"))
    {
        const auto& velocity = jsonFile["velocity"];

        if (velocity.contains("split") && velocity["split"].contains("minimalSpeedThreshold"))
        {
            params.velocity.split.minimalSpeedThreshold = velocity["split"]["minimalSpeedThreshold"].get<float>();
        }

        if (velocity.contains("merge") && velocity["merge"].contains("maximalSpeedThreshold"))
        {
            params.velocity.merge.maximalSpeedThreshold = velocity["merge"]["maximalSpeedThreshold"].get<float>();
        }
    }

    if (jsonFile.contains("interface"))
    {
        const auto& interface = jsonFile["interface"];

        if (interface.contains("split") && interface["split"].contains("distanceRatioThreshold"))
        {
            params.interfaceParameters.split.distanceRatioThreshold =
                interface["split"]["distanceRatioThreshold"].get<float>();
        }

        if (interface.contains("merge") && interface["merge"].contains("distanceRatioThreshold"))
        {
            params.interfaceParameters.merge.distanceRatioThreshold =
                interface["merge"]["distanceRatioThreshold"].get<float>();
        }
    }

    if (jsonFile.contains("vorticity"))
    {
        const auto& vorticity = jsonFile["vorticity"];
        if (vorticity.contains("split") && vorticity["split"].contains("minimalVorticityThreshold"))
        {
            params.vorticity.split.minimalVorticityThreshold =
                vorticity["split"]["minimalVorticityThreshold"].get<float>();
        }
        if (vorticity.contains("merge") && vorticity["merge"].contains("maximalVorticityThreshold"))
        {
            params.vorticity.merge.maximalVorticityThreshold =
                vorticity["merge"]["maximalVorticityThreshold"].get<float>();
        }
    }

    if (jsonFile.contains("curvature"))
    {
        const auto& curvature = jsonFile["curvature"];
        if (curvature.contains("split") && curvature["split"].contains("minimalCurvatureThreshold"))
        {
            params.curvature.split.minimalCurvatureThreshold =
                curvature["split"]["minimalCurvatureThreshold"].get<float>();
        }
        if (curvature.contains("merge") && curvature["merge"].contains("maximalCurvatureThreshold"))
        {
            params.curvature.merge.maximalCurvatureThreshold =
                curvature["merge"]["maximalCurvatureThreshold"].get<float>();
        }
    }

    _refinementParams = params;
}

//NOLINTNEXTLINE(readability-function-cognitive-complexity)
void ConfigurationManager::parseInitialParameters(const json& jsonFile)
{
    InitialParameters params;

    if (jsonFile.contains("liquidVolumeFraction"))
    {
        params.liquidVolumeFraction = jsonFile["liquidVolumeFraction"].get<float>();
    }

    if (jsonFile.contains("particleCount"))
    {
        if (jsonFile["particleCount"].is_array() && jsonFile["particleCount"].size() == 3)
        {
            params.particleCount = glm::uvec3(jsonFile["particleCount"][0].get<uint32_t>(),
                                              jsonFile["particleCount"][1].get<uint32_t>(),
                                              jsonFile["particleCount"][2].get<uint32_t>());
        }
        else if (jsonFile["particleCount"].is_number())
        {
            const auto total = jsonFile["particleCount"].get<uint32_t>();
            const auto cubeRoot = std::pow(total, 1.F / 3.F);
            const auto perAxis = static_cast<uint32_t>(std::ceil(cubeRoot));
            params.particleCount = glm::uvec3(perAxis, perAxis, perAxis);
            panda::log::Warning("Converting single particleCount value to 3D: {}x{}x{}", perAxis, perAxis, perAxis);
        }
    }

    _initialParams = params;
}

//NOLINTNEXTLINE(readability-function-cognitive-complexity)
void ConfigurationManager::parseBenchmarkParameters(const json& jsonFile)
{
    BenchmarkParameters params;

    if (jsonFile.contains("enabled"))
    {
        params.enabled = jsonFile["enabled"].get<bool>();
    }

    if (jsonFile.contains("testCase"))
    {
        params.testCase = jsonFile["testCase"].get<std::string>();
    }

    if (jsonFile.contains("outputPath"))
    {
        params.outputPath = jsonFile["outputPath"].get<std::string>();
    }

    if (jsonFile.contains("reynoldsNumber"))
    {
        params.reynoldsNumber = jsonFile["reynoldsNumber"].get<float>();
    }

    if (jsonFile.contains("simulations"))
    {
        const auto& simulations = jsonFile["simulations"];
        if (simulations.contains("coarse") && simulations["coarse"].contains("particleSize"))
        {
            params.coarse.particleSize = simulations["coarse"]["particleSize"].get<float>();
        }
        if (simulations.contains("fine") && simulations["fine"].contains("particleSize"))
        {
            params.fine.particleSize = simulations["fine"]["particleSize"].get<float>();
        }
        if (simulations.contains("adaptive"))
        {
            const auto& adaptive = simulations["adaptive"];

            if (adaptive.contains("minParticleSize"))
            {
                params.adaptive.minParticleSize = adaptive["minParticleSize"].get<float>();
            }

            if (adaptive.contains("maxParticleSize"))
            {
                params.adaptive.maxParticleSize = adaptive["maxParticleSize"].get<float>();
            }
        }
    }

    if (jsonFile.contains("measurementInterval"))
    {
        params.measurementInterval = jsonFile["measurementInterval"].get<uint32_t>();
    }

    if (jsonFile.contains("totalSimulationFrames"))
    {
        params.totalSimulationFrames = jsonFile["totalSimulationFrames"].get<uint32_t>();
    }

    // Test case specific parameters
    if (jsonFile.contains("poiseuille"))
    {
        const auto& poiseuille = jsonFile["poiseuille"];
        if (poiseuille.contains("channelHeight"))
        {
            params.channelHeight = poiseuille["channelHeight"].get<float>();
        }
        if (poiseuille.contains("channelLength"))
        {
            params.channelLength = poiseuille["channelLength"].get<float>();
        }
        if (poiseuille.contains("channelWidth"))
        {
            params.channelWidth = poiseuille["channelWidth"].get<float>();
        }
    }

    if (jsonFile.contains("taylorGreen"))
    {
        const auto& taylorGreen = jsonFile["taylorGreen"];
        if (taylorGreen.contains("domainSize"))
        {
            params.domainSize = taylorGreen["domainSize"].get<float>();
        }
    }

    if (jsonFile.contains("damBreak"))
    {
        const auto& damBreak = jsonFile["damBreak"];
        if (damBreak.contains("tankLength"))
        {
            params.tankLength = damBreak["tankLength"].get<float>();
        }
        if (damBreak.contains("tankHeight"))
        {
            params.tankHeight = damBreak["tankHeight"].get<float>();
        }
        if (damBreak.contains("tankWidth"))
        {
            params.tankWidth = damBreak["tankWidth"].get<float>();
        }
        if (damBreak.contains("waterColumnWidth"))
        {
            params.waterColumnWidth = damBreak["waterColumnWidth"].get<float>();
        }
        if (damBreak.contains("waterColumnHeight"))
        {
            params.waterColumnHeight = damBreak["waterColumnHeight"].get<float>();
        }
    }

    if (jsonFile.contains("lidDrivenCavity"))
    {
        const auto& lidDrivenCavity = jsonFile["lidDrivenCavity"];
        if (lidDrivenCavity.contains("cavitySize"))
        {
            params.cavitySize = lidDrivenCavity["cavitySize"].get<float>();
        }
    }

    _benchmarkParams = params;
}

auto ConfigurationManager::getSimulationParameters() const -> std::optional<cuda::Simulation::Parameters>
{
    return _simulationParams;
}

auto ConfigurationManager::getRefinementParameters() const -> std::optional<cuda::refinement::RefinementParameters>
{
    return _refinementParams;
}

auto ConfigurationManager::getInitialParameters() const -> std::optional<InitialParameters>
{
    return _initialParams;
}

auto ConfigurationManager::getBenchmarkParameters() const -> std::optional<BenchmarkParameters>
{
    return _benchmarkParams;
}

}
