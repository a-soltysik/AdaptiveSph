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
#include "glm/ext/scalar_constants.hpp"

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
void ConfigurationManager::parseRefinementParameters(const json& jsonFile)
{
    cuda::refinement::RefinementParameters params;

    params.enabled = parseScalarProperty(jsonFile, "enabled", false);
    params.maxParticleCount = parseScalarProperty(jsonFile, "maxParticleCount", 1000000);
    params.maxBatchRatio = parseScalarProperty(jsonFile, "maxBatchRatio", 0.5f);
    params.minMassRatio = parseScalarProperty(jsonFile, "minMassRatio", 0.01f);
    params.maxMassRatio = parseScalarProperty(jsonFile, "maxMassRatio", 100.0f);
    params.initialCooldown = parseScalarProperty(jsonFile, "initialCooldown", 1000);
    params.cooldown = parseScalarProperty(jsonFile, "cooldown", 1000);
    params.criterionType = parseScalarProperty(jsonFile, "criterionType", std::string {"interface"});

    if (jsonFile.contains("splitting"))
    {
        const auto& splitting = jsonFile["splitting"];
        params.splitting.epsilon = parseScalarProperty(splitting, "epsilon", 0.6f);
        params.splitting.alpha = parseScalarProperty(splitting, "alpha", 0.6f);
        params.splitting.centerMassRatio = parseScalarProperty(splitting, "centerMassRatio", 0.2f);
        params.splitting.vertexMassRatio = parseScalarProperty(splitting, "vertexMassRatio", 0.067f);
    }

    if (jsonFile.contains("velocity"))
    {
        const auto& velocity = jsonFile["velocity"];

        params.velocity.split.minimalSpeedThreshold =
            parseScalarProperty(velocity["split"], "minimalSpeedThreshold", 0.5f);
        params.velocity.merge.maximalSpeedThreshold =
            parseScalarProperty(velocity["merge"], "maximalSpeedThreshold", 4.0f);
    }

    if (jsonFile.contains("interface"))
    {
        const auto& interface = jsonFile["interface"];

        params.interfaceParameters.split.distanceRatioThreshold =
            parseScalarProperty(interface["split"], "distanceRatioThreshold", 0.07f);
        params.interfaceParameters.merge.distanceRatioThreshold =
            parseScalarProperty(interface["merge"], "distanceRatioThreshold", 0.18f);
    }

    if (jsonFile.contains("vorticity"))
    {
        const auto& vorticity = jsonFile["vorticity"];

        params.vorticity.split.minimalVorticityThreshold =
            parseScalarProperty(vorticity["split"], "minimalVorticityThreshold", 34.f);
        params.vorticity.merge.maximalVorticityThreshold =
            parseScalarProperty(vorticity["merge"], "maximalVorticityThreshold", 3.4F);
    }

    if (jsonFile.contains("curvature"))
    {
        const auto& curvature = jsonFile["curvature"];
        params.curvature.split.minimalCurvatureThreshold =
            parseScalarProperty(curvature["split"], "minimalCurvatureThreshold", 25000.F);
        params.curvature.merge.maximalCurvatureThreshold =
            parseScalarProperty(curvature["merge"], "maximalCurvatureThreshold", 12800.F);
    }

    _refinementParams = params;
}

void ConfigurationManager::parseInitialParameters(const json& jsonFile)
{
    InitialParameters params;

    params.particleCount = parseVec3Property(jsonFile, "particleCount", glm::uvec3 {20, 20, 20});

    _initialParams = params;
}

//NOLINTNEXTLINE(readability-function-cognitive-complexity)
void ConfigurationManager::parseBenchmarkParameters(const json& jsonFile)
{
    BenchmarkParameters params;

    params.enabled = parseScalarProperty(jsonFile, "enabled", false);

    if (jsonFile.contains("testCase"))
    {
        if (jsonFile["testCase"].get<std::string>() == "lidDrivenCavity")
        {
            params.testCase = cuda::Simulation::Parameters::TestCase::LidDrivenCavity;
        }
        else if (jsonFile["testCase"].get<std::string>() == "poiseuilleFlow")
        {
            params.testCase = cuda::Simulation::Parameters::TestCase::PoiseuilleFlow;
        }
        else if (jsonFile["testCase"].get<std::string>() == "taylorGreenVortex")
        {
            params.testCase = cuda::Simulation::Parameters::TestCase::TaylorGreenVortex;
        }
    }

    params.outputPath = parseScalarProperty(jsonFile, "outputPath", std::string {"output"});

    // Parse resolution-specific parameters
    if (jsonFile.contains("resolutions"))
    {
        const auto& resolutions = jsonFile["resolutions"];

        // Parse Coarse simulation parameters
        if (resolutions.contains("coarse"))
        {
            const auto& coarse = resolutions["coarse"];
            params.coarse.baseParticleRadius = parseScalarProperty(coarse, "baseParticleRadius", 0.25F);
            params.coarse.baseParticleMass = parseScalarProperty(coarse, "baseParticleMass", 1.F);
            params.coarse.baseSmoothingRadius = parseScalarProperty(coarse, "baseSmoothingRadius", 1.F);
            params.coarse.pressureConstant = parseScalarProperty(coarse, "pressureConstant", 1.F);
            params.coarse.nearPressureConstant = parseScalarProperty(coarse, "nearPressureConstant", 0.01F);
            params.coarse.viscosityConstant = parseScalarProperty(coarse, "viscosityConstant", 0.001F);
        }
        // Parse Fine simulation parameters
        if (resolutions.contains("fine"))
        {
            const auto& fine = resolutions["fine"];
            params.fine.baseParticleRadius = parseScalarProperty(fine, "baseParticleRadius", 0.25F);
            params.fine.baseParticleMass = parseScalarProperty(fine, "baseParticleMass", 1.F);
            params.fine.baseSmoothingRadius = parseScalarProperty(fine, "baseSmoothingRadius", 1.F);
            params.fine.pressureConstant = parseScalarProperty(fine, "pressureConstant", 1.F);
            params.fine.nearPressureConstant = parseScalarProperty(fine, "nearPressureConstant", 0.01F);
            params.fine.viscosityConstant = parseScalarProperty(fine, "viscosityConstant", 0.001F);
        }
        // Parse Adaptive simulation parameters
        if (resolutions.contains("adaptive"))
        {
            const auto& adaptive = resolutions["adaptive"];
            params.adaptive.baseParticleRadius = parseScalarProperty(adaptive, "baseParticleRadius", 0.25F);
            params.adaptive.baseParticleMass = parseScalarProperty(adaptive, "baseParticleMass", 1.F);
            params.adaptive.baseSmoothingRadius = parseScalarProperty(adaptive, "baseSmoothingRadius", 1.F);
            params.adaptive.pressureConstant = parseScalarProperty(adaptive, "pressureConstant", 1.F);
            params.adaptive.nearPressureConstant = parseScalarProperty(adaptive, "nearPressureConstant", 0.01F);
            params.adaptive.viscosityConstant = parseScalarProperty(adaptive, "viscosityConstant", 0.001F);
        }
    }

    params.measurementInterval = parseScalarProperty(jsonFile, "measurementInterval", 1000);
    params.totalSimulationFrames = parseScalarProperty(jsonFile, "totalSimulationFrames", 100000);
    params.timestep = parseScalarProperty(jsonFile, "timestep", 0.001F);

    if (jsonFile.contains("poiseuille"))
    {
        const auto& poiseuille = jsonFile["poiseuille"];
        params.channelHeight = parseScalarProperty(poiseuille, "channelHeight", 1.F);
        params.channelLength = parseScalarProperty(poiseuille, "channelLength", 5.F);
        params.channelWidth = parseScalarProperty(poiseuille, "channelWidth", 1.F);
        params.forceMagnitude = parseScalarProperty(poiseuille, "forceMagnitude", 5.F);
    }

    if (jsonFile.contains("taylorGreen"))
    {
        const auto& taylorGreen = jsonFile["taylorGreen"];
        params.domainSize = parseScalarProperty(taylorGreen, "domainSize", 2.F * glm::pi<float>());
    }

    if (jsonFile.contains("lidDrivenCavity"))
    {
        const auto& lidDrivenCavity = jsonFile["lidDrivenCavity"];
        params.cavitySize = parseScalarProperty(lidDrivenCavity, "cavitySize", 3.F);
        params.lidVelocity = parseScalarProperty(lidDrivenCavity, "lidVelocity", 1.F);
    }

    if (_refinementParams.has_value())
    {
        params.refinement = _refinementParams.value();
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

void ConfigurationManager::parseSimulationParameters(const json& jsonFile)
{
    cuda::Simulation::Parameters params {};
    parseDomainParameters(jsonFile, params);
    parseFluidParameters(jsonFile, params);
    parseSimulationControlParameters(jsonFile, params);
    _simulationParams = params;
}

void ConfigurationManager::parseDomainParameters(const json& jsonFile, cuda::Simulation::Parameters& params)
{
    if (!jsonFile.contains("domain"))
    {
        return;
    }

    const auto& domain = jsonFile["domain"];
    params.domain.min = parseVec3Property(domain, "min", glm::vec3(0.0f));
    params.domain.max = parseVec3Property(domain, "max", glm::vec3(1.0f));
}

void ConfigurationManager::parseFluidParameters(const json& jsonFile, cuda::Simulation::Parameters& params)
{
    params.restDensity = parseScalarProperty(jsonFile, "restDensity", 1000.0f);
    params.pressureConstant = parseScalarProperty(jsonFile, "pressureConstant", 100.0f);
    params.nearPressureConstant = parseScalarProperty(jsonFile, "nearPressureConstant", 100.0f);
    params.viscosityConstant = parseScalarProperty(jsonFile, "viscosityConstant", 0.01f);
}

void ConfigurationManager::parseSimulationControlParameters(const json& jsonFile, cuda::Simulation::Parameters& params)
{
    params.gravity = parseVec3Property(jsonFile, "gravity", glm::vec3(0.0f, -9.81f, 0.0f));
    params.restitution = parseScalarProperty(jsonFile, "restitution", 0.5f);
    params.maxVelocity = parseScalarProperty(jsonFile, "maxVelocity", 100.0f);
    params.baseParticleRadius = parseScalarProperty(jsonFile, "baseParticleRadius", 0.01f);
    params.baseParticleMass = parseScalarProperty(jsonFile, "baseParticleMass", 1.0f);
    params.baseSmoothingRadius = parseScalarProperty(jsonFile, "baseSmoothingRadius", 0.02f);
    params.threadsPerBlock = parseScalarProperty<uint32_t>(jsonFile, "threadsPerBlock", 256);
}

template <typename T>
auto ConfigurationManager::parseScalarProperty(const json& jsonFile, const std::string& propertyName, T defaultValue)
    -> T
{
    if (jsonFile.contains(propertyName))
    {
        return jsonFile[propertyName].get<T>();
    }
    return defaultValue;
}

template <typename T>
auto ConfigurationManager::parseVec3Property(const json& jsonFile, const std::string& propertyName, T defaultValue) -> T
{
    if (jsonFile.contains(propertyName) && jsonFile[propertyName].is_array() && jsonFile[propertyName].size() == 3)
    {
        return T(jsonFile[propertyName][0].get<typename T::value_type>(),
                 jsonFile[propertyName][1].get<typename T::value_type>(),
                 jsonFile[propertyName][2].get<typename T::value_type>());
    }
    return defaultValue;
}

}
