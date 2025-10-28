#include "ConfigurationManager.hpp"

#include <panda/Logger.h>

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
    params.maxParticleCount = parseScalarProperty(jsonFile, "maxParticleCount", uint32_t {1000000});
    params.maxBatchRatio = parseScalarProperty(jsonFile, "maxBatchRatio", 0.5F);
    params.minMassRatio = parseScalarProperty(jsonFile, "minMassRatio", 0.01F);
    params.maxMassRatio = parseScalarProperty(jsonFile, "maxMassRatio", 100.0F);
    params.initialCooldown = parseScalarProperty(jsonFile, "initialCooldown", uint32_t {1000});
    params.cooldown = parseScalarProperty(jsonFile, "cooldown", uint32_t {1000});

    if (jsonFile.contains("criterionType"))
    {
        if (jsonFile["criterionType"].get<std::string>() == "velocity")
        {
            params.criterionType = cuda::refinement::RefinementParameters::Criterion::Velocity;
        }
        if (jsonFile["criterionType"].get<std::string>() == "vorticity")
        {
            params.criterionType = cuda::refinement::RefinementParameters::Criterion::Vorticity;
        }
        if (jsonFile["criterionType"].get<std::string>() == "curvature")
        {
            params.criterionType = cuda::refinement::RefinementParameters::Criterion::Curvature;
        }
        if (jsonFile["criterionType"].get<std::string>() == "interface")
        {
            params.criterionType = cuda::refinement::RefinementParameters::Criterion::Interface;
        }
    }

    if (jsonFile.contains("splitting"))
    {
        const auto& splitting = jsonFile["splitting"];
        params.splitting.epsilon = parseScalarProperty(splitting, "epsilon", 0.6F);
        params.splitting.alpha = parseScalarProperty(splitting, "alpha", 0.6F);
        params.splitting.centerMassRatio = parseScalarProperty(splitting, "centerMassRatio", 0.2F);
        params.splitting.vertexMassRatio = parseScalarProperty(splitting, "vertexMassRatio", 0.067F);
    }

    if (jsonFile.contains("velocity"))
    {
        const auto& velocity = jsonFile["velocity"];

        params.velocity.split.minimalSpeedThreshold =
            parseScalarProperty(velocity["split"], "minimalSpeedThreshold", 0.5F);
        params.velocity.merge.maximalSpeedThreshold =
            parseScalarProperty(velocity["merge"], "maximalSpeedThreshold", 4.0F);
    }

    if (jsonFile.contains("interface"))
    {
        const auto& interface = jsonFile["interface"];

        params.interfaceParameters.split.distanceRatioThreshold =
            parseScalarProperty(interface["split"], "distanceRatioThreshold", 0.07F);
        params.interfaceParameters.merge.distanceRatioThreshold =
            parseScalarProperty(interface["merge"], "distanceRatioThreshold", 0.18F);
    }

    if (jsonFile.contains("vorticity"))
    {
        const auto& vorticity = jsonFile["vorticity"];

        params.vorticity.split.minimalVorticityThreshold =
            parseScalarProperty(vorticity["split"], "minimalVorticityThreshold", 34.F);
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
    params.domain.min = parseVec3Property(domain, "min", glm::vec3 {});
    params.domain.max = parseVec3Property(domain, "max", glm::vec3 {1.0F});
}

void ConfigurationManager::parseFluidParameters(const json& jsonFile, cuda::Simulation::Parameters& params)
{
    params.restDensity = parseScalarProperty(jsonFile, "restDensity", 1000.0F);
    params.pressureConstant = parseScalarProperty(jsonFile, "pressureConstant", 100.0F);
    params.nearPressureConstant = parseScalarProperty(jsonFile, "nearPressureConstant", 100.0F);
    params.viscosityConstant = parseScalarProperty(jsonFile, "viscosityConstant", 0.01F);
}

void ConfigurationManager::parseSimulationControlParameters(const json& jsonFile, cuda::Simulation::Parameters& params)
{
    params.gravity = parseVec3Property(jsonFile, "gravity", glm::vec3(0.0F, 9.81F, 0.0F));
    params.restitution = parseScalarProperty(jsonFile, "restitution", 0.5F);
    params.maxVelocity = parseScalarProperty(jsonFile, "maxVelocity", 100.0F);
    params.baseParticleRadius = parseScalarProperty(jsonFile, "baseParticleRadius", 0.01F);
    params.baseParticleMass = parseScalarProperty(jsonFile, "baseParticleMass", 1.0F);
    params.baseSmoothingRadius = parseScalarProperty(jsonFile, "baseSmoothingRadius", 0.02F);
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
