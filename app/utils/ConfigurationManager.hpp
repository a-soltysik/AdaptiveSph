#pragma once

#include <cuda/Simulation.cuh>
#include <cuda/refinement/RefinementParameters.cuh>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>

#include "glm/vec3.hpp"

namespace sph
{

struct InitialParameters
{
    float liquidVolumeFraction = 0.1F;
    glm::uvec3 particleCount = {40, 40, 25};
};

class ConfigurationManager
{
public:
    ConfigurationManager() = default;
    ~ConfigurationManager() = default;
    auto loadFromFile(const std::string& filePath) -> bool;
    auto loadFromString(const std::string& jsonString) -> bool;
    [[nodiscard]] std::optional<cuda::Simulation::Parameters> getSimulationParameters() const;
    [[nodiscard]] std::optional<cuda::refinement::RefinementParameters> getRefinementParameters() const;
    [[nodiscard]] std::optional<InitialParameters> getInitialParameters() const;

private:
    void parseSimulationParameters(const nlohmann::json& j);
    void parseRefinementParameters(const nlohmann::json& j);
    void parseInitialParameters(const nlohmann::json& j);

    std::optional<cuda::Simulation::Parameters> _simulationParams;
    std::optional<cuda::refinement::RefinementParameters> _refinementParams;
    std::optional<InitialParameters> _initialParams;
};

}  // namespace sph
