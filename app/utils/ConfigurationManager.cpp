#include "ConfigurationManager.hpp"

#include <panda/Logger.h>

#include <cuda/refinement/VelocityParameters.cuh>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace sph
{

using json = nlohmann::json;

bool ConfigurationManager::loadFromFile(const std::string& filePath)
{
    try
    {
        std::ifstream file(filePath);
        if (!file.is_open())
        {
            panda::log::Error("Failed to open configuration file: {}", filePath);
            return false;
        }

        json j;
        file >> j;

        if (j.contains("simulation"))
        {
            parseSimulationParameters(j["simulation"]);
        }

        if (j.contains("refinement"))
        {
            parseRefinementParameters(j["refinement"]);
        }

        if (j.contains("initial"))
        {
            parseInitialParameters(j["initial"]);
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

bool ConfigurationManager::loadFromString(const std::string& jsonString)
{
    try
    {
        json j = json::parse(jsonString);

        if (j.contains("simulation"))
        {
            parseSimulationParameters(j["simulation"]);
        }

        if (j.contains("refinement"))
        {
            parseRefinementParameters(j["refinement"]);
        }

        if (j.contains("initial"))
        {
            parseInitialParameters(j["initial"]);
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

void ConfigurationManager::parseSimulationParameters(const json& j)
{
    cuda::Simulation::Parameters params;

    if (j.contains("domain"))
    {
        auto& domain = j["domain"];
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

    if (j.contains("gravity") && j["gravity"].is_array() && j["gravity"].size() == 3)
    {
        params.gravity =
            glm::vec3(j["gravity"][0].get<float>(), j["gravity"][1].get<float>(), j["gravity"][2].get<float>());
    }

    if (j.contains("restDensity"))
    {
        params.restDensity = j["restDensity"].get<float>();
    }

    if (j.contains("pressureConstant"))
    {
        params.pressureConstant = j["pressureConstant"].get<float>();
    }

    if (j.contains("nearPressureConstant"))
    {
        params.nearPressureConstant = j["nearPressureConstant"].get<float>();
    }

    if (j.contains("restitution"))
    {
        params.restitution = j["restitution"].get<float>();
    }

    if (j.contains("baseSmoothingRadius"))
    {
        params.baseSmoothingRadius = j["baseSmoothingRadius"].get<float>();
    }

    if (j.contains("viscosityConstant"))
    {
        params.viscosityConstant = j["viscosityConstant"].get<float>();
    }

    if (j.contains("maxVelocity"))
    {
        params.maxVelocity = j["maxVelocity"].get<float>();
    }

    if (j.contains("baseParticleRadius"))
    {
        params.baseParticleRadius = j["baseParticleRadius"].get<float>();
    }

    if (j.contains("baseParticleMass"))
    {
        params.baseParticleMass = j["baseParticleMass"].get<float>();
    }

    if (j.contains("threadsPerBlock"))
    {
        params.threadsPerBlock = j["threadsPerBlock"].get<uint32_t>();
    }

    _simulationParams = params;
}

void ConfigurationManager::parseRefinementParameters(const json& j)
{
    cuda::refinement::RefinementParameters params;

    if (j.contains("enabled"))
    {
        params.enabled = j["enabled"].get<bool>();
    }

    if (j.contains("minMassRatio"))
    {
        params.minMassRatio = j["minMassRatio"].get<float>();
    }

    if (j.contains("maxMassRatio"))
    {
        params.maxMassRatio = j["maxMassRatio"].get<float>();
    }

    if (j.contains("maxParticleCount"))
    {
        params.maxParticleCount = j["maxParticleCount"].get<uint32_t>();
    }

    if (j.contains("maxBatchRatio"))
    {
        params.maxBatchRatio = j["maxBatchRatio"].get<float>();
    }

    if (j.contains("initialCooldown"))
    {
        params.initialCooldown = j["initialCooldown"].get<uint32_t>();
    }

    if (j.contains("cooldown"))
    {
        params.cooldown = j["cooldown"].get<uint32_t>();
    }

    if (j.contains("splitting"))
    {
        auto& splitting = j["splitting"];

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

    if (j.contains("velocity"))
    {
        auto& velocity = j["velocity"];

        if (velocity.contains("split") && velocity["split"].contains("minimalSpeedThreshold"))
        {
            params.velocity.split.minimalSpeedThreshold = velocity["split"]["minimalSpeedThreshold"].get<float>();
        }

        if (velocity.contains("merge") && velocity["merge"].contains("maximalSpeedThreshold"))
        {
            params.velocity.merge.maximalSpeedThreshold = velocity["merge"]["maximalSpeedThreshold"].get<float>();
        }
    }

    _refinementParams = params;
}

void ConfigurationManager::parseInitialParameters(const json& j)
{
    InitialParameters params;

    if (j.contains("liquidVolumeFraction"))
    {
        params.liquidVolumeFraction = j["liquidVolumeFraction"].get<float>();
    }

    if (j.contains("particleCount"))
    {
        if (j["particleCount"].is_array() && j["particleCount"].size() == 3)
        {
            // Parse as 3D vector
            params.particleCount = glm::uvec3(j["particleCount"][0].get<uint32_t>(),
                                              j["particleCount"][1].get<uint32_t>(),
                                              j["particleCount"][2].get<uint32_t>());
        }
        else if (j["particleCount"].is_number())
        {
            // For backward compatibility - convert single number to cubed root distribution
            uint32_t total = j["particleCount"].get<uint32_t>();
            float cubeRoot = std::pow(total, 1.0f / 3.0f);
            uint32_t perAxis = static_cast<uint32_t>(std::ceil(cubeRoot));
            params.particleCount = glm::uvec3(perAxis, perAxis, perAxis);
            panda::log::Warning("Converting single particleCount value to 3D: {}x{}x{}", perAxis, perAxis, perAxis);
        }
    }

    _initialParams = params;
}

std::optional<cuda::Simulation::Parameters> ConfigurationManager::getSimulationParameters() const
{
    return _simulationParams;
}

std::optional<cuda::refinement::RefinementParameters> ConfigurationManager::getRefinementParameters() const
{
    return _refinementParams;
}

std::optional<InitialParameters> ConfigurationManager::getInitialParameters() const
{
    return _initialParams;
}

}  // namespace sph
