#pragma once

#include <cstdint>
#include <cuda/Simulation.cuh>
#include <cuda/refinement/RefinementParameters.cuh>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_uint3.hpp>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <string>

#include "cuda/Simulation.cuh"

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
    cuda::Simulation::Parameters::TestCase testCase = cuda::Simulation::Parameters::TestCase::LidDrivenCavity;
    std::string outputPath = "benchmarks/";

    struct SimulationConfig
    {
        float particleSize = 0.05F;
    };

    struct AdaptiveConfig
    {
        float minParticleSize = 0.025F;
        float maxParticleSize = 0.05F;
    };

    SimulationConfig coarse;
    SimulationConfig fine;
    AdaptiveConfig adaptive;
    uint32_t measurementInterval = 20;
    uint32_t totalSimulationFrames = 1000;
    // Test case specific parameters
    float reynoldsNumber = 100.0F;
    // Poiseuille flow parameters
    float channelHeight = 0.1F;
    float channelLength = 0.5F;
    float channelWidth = 0.1F;
    // Taylor-Green parameters
    float domainSize = 1.0F;

    // Dam break parameters
    float tankLength = 4.0F;
    float tankHeight = 2.0F;
    float tankWidth = 1.0F;
    float waterColumnWidth = 1.0F;
    float waterColumnHeight = 1.0F;

    // Lid-driven cavity parameters
    float cavitySize = 1.0F;
};

class ConfigurationManager
{
public:
    auto loadFromFile(const std::string& filePath) -> bool;
    auto loadFromString(const std::string& jsonString) -> bool;
    [[nodiscard]] auto getSimulationParameters() const -> std::optional<cuda::Simulation::Parameters>;
    [[nodiscard]] auto getRefinementParameters() const -> std::optional<cuda::refinement::RefinementParameters>;
    [[nodiscard]] auto getInitialParameters() const -> std::optional<InitialParameters>;
    [[nodiscard]] auto getBenchmarkParameters() const -> std::optional<BenchmarkParameters>;

private:
    void parseSimulationParameters(const nlohmann::json& jsonFile);
    void parseRefinementParameters(const nlohmann::json& jsonFile);
    void parseInitialParameters(const nlohmann::json& jsonFile);
    void parseBenchmarkParameters(const nlohmann::json& jsonFile);

    std::optional<cuda::Simulation::Parameters> _simulationParams;
    std::optional<cuda::refinement::RefinementParameters> _refinementParams;
    std::optional<InitialParameters> _initialParams;
    std::optional<BenchmarkParameters> _benchmarkParams;
};

}
