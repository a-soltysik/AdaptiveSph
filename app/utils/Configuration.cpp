#include "Configuration.hpp"

#include <panda/Logger.h>

#include <../../cuda/include/cuda/simulation/Simulation.cuh>
#include <cuda/refinement/RefinementParameters.cuh>
#include <exception>
#include <fstream>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <string>

#include "utils/Serializers.hpp"  //NOLINT(misc-include-cleaner)

using json = nlohmann::json;

namespace sph::utils
{

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

auto dumpTemplateConfiguration(const std::string& filePath) -> void
{
    try
    {
        std::ofstream file(filePath);
        if (!file.is_open())
        {
            panda::log::Error("Failed to open file: {}", filePath);
        }

        const json jsonFile = Configuration {.initialParameters = InitialParameters {},
                                             .simulationParameters = cuda::Simulation::Parameters {},
                                             .refinementParameters = cuda::refinement::RefinementParameters {},
                                             .renderingParameters = RenderingParameters {}};
        file << jsonFile.dump(4);

        panda::log::Info("Configuration file saved: {}", filePath);
    }
    catch (const json::exception& e)
    {
        panda::log::Error("JSON parsing error: {}", e.what());
    }
    catch (const std::exception& e)
    {
        panda::log::Error("Error saving configuration: {}", e.what());
    }
}

}
