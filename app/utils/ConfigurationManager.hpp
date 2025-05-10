// ConfigurationManager.hpp
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

struct BenchmarkParameters
{
    bool enabled = false;
    std::string testCase = "lidDrivenCavity";
    std::string outputPath = "benchmarks/";

    struct SimulationConfig
    {
        float particleSize = 0.05f;
    };

    struct AdaptiveConfig
    {
        float minParticleSize = 0.025f;
        float maxParticleSize = 0.05f;
    };

    SimulationConfig coarse;
    SimulationConfig fine;
    AdaptiveConfig adaptive;
    uint32_t measurementInterval = 20;
    uint32_t totalSimulationFrames = 1000;
    // Test case specific parameters
    float reynoldsNumber = 100.0f;
    // Poiseuille flow parameters
    float channelHeight = 0.1f;
    float channelLength = 0.5f;
    float channelWidth = 0.1f;
    // Taylor-Green parameters
    float domainSize = 1.0f;

    // Dam break parameters
    float tankLength = 4.0f;
    float tankHeight = 2.0f;
    float tankWidth = 1.0f;
    float waterColumnWidth = 1.0f;
    float waterColumnHeight = 1.0f;

    // Lid-driven cavity parameters
    float cavitySize = 1.0f;
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
    [[nodiscard]] std::optional<BenchmarkParameters> getBenchmarkParameters() const;

private:
    void parseSimulationParameters(const nlohmann::json& j);
    void parseRefinementParameters(const nlohmann::json& j);
    void parseInitialParameters(const nlohmann::json& j);
    void parseBenchmarkParameters(const nlohmann::json& j);

    std::optional<cuda::Simulation::Parameters> _simulationParams;
    std::optional<cuda::refinement::RefinementParameters> _refinementParams;
    std::optional<InitialParameters> _initialParams;
    std::optional<BenchmarkParameters> _benchmarkParams;
};

}
