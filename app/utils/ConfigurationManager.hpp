#pragma once

#include <cuda/Simulation.cuh>
#include <cuda/refinement/RefinementParameters.cuh>
#include <glm/ext/vector_uint3.hpp>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <string>

namespace sph
{
struct InitialParameters
{
    glm::uvec3 particleCount = {40, 40, 25};
};

struct SimulationResolutionConfig
{
    float baseParticleRadius = 0.025F;
    float baseParticleMass = 1.2F;
    float baseSmoothingRadius = 0.22F;
    float pressureConstant = 0.5F;
    float nearPressureConstant = 0.1F;
    float viscosityConstant = 0.001F;
};

class ConfigurationManager
{
public:
    auto loadFromFile(const std::string& filePath) -> bool;

    auto loadFromString(const std::string& jsonString) -> bool;

    [[nodiscard]] auto getSimulationParameters() const -> std::optional<cuda::Simulation::Parameters>;

    [[nodiscard]] auto getRefinementParameters() const -> std::optional<cuda::refinement::RefinementParameters>;

    [[nodiscard]] auto getInitialParameters() const -> std::optional<InitialParameters>;

private:
    void parseSimulationParameters(const nlohmann::json& jsonFile);

    void parseRefinementParameters(const nlohmann::json& jsonFile);

    void parseInitialParameters(const nlohmann::json& jsonFile);

    static void parseDomainParameters(const nlohmann::json& jsonFile, cuda::Simulation::Parameters& params);

    static void parseFluidParameters(const nlohmann::json& jsonFile, cuda::Simulation::Parameters& params);

    static void parseSimulationControlParameters(const nlohmann::json& jsonFile, cuda::Simulation::Parameters& params);

    template <typename T>
    static auto parseScalarProperty(const nlohmann::json& jsonFile, const std::string& propertyName, T defaultValue)
        -> T;

    template <typename T>
    static auto parseVec3Property(const nlohmann::json& jsonFile, const std::string& propertyName, T defaultValue) -> T;

    std::optional<cuda::Simulation::Parameters> _simulationParams;
    std::optional<cuda::refinement::RefinementParameters> _refinementParams;
    std::optional<InitialParameters> _initialParams;
};
}
