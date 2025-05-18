#pragma once

#include <cstdint>
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

struct BenchmarkParameters
{
    bool enabled = false;
    cuda::Simulation::Parameters::TestCase testCase = cuda::Simulation::Parameters::TestCase::LidDrivenCavity;
    std::string outputPath = "benchmarks/";
    SimulationResolutionConfig coarse;
    SimulationResolutionConfig fine;
    SimulationResolutionConfig adaptive;

    uint32_t measurementInterval = 20;
    uint32_t totalSimulationFrames = 1000;
    float timestep = 0.0001F;  // Simulation time step
    // Poiseuille flow parameters
    float channelHeight = 0.1F;
    float channelLength = 0.5F;
    float channelWidth = 0.1F;
    float forceMagnitude = 10.0F;  // Added for Poiseuille flow

    // Taylor-Green parameters
    float domainSize = 6.28F;

    // Dam break parameters (kept for backward compatibility)
    float tankLength = 4.0F;
    float tankHeight = 2.0F;
    float tankWidth = 1.0F;
    float waterColumnWidth = 1.0F;
    float waterColumnHeight = 1.0F;

    // Lid-driven cavity parameters
    float cavitySize = 1.0F;
    float lidVelocity = 5.0F;  // Added for Lid-Driven Cavity

    // Refinement parameters (reference to the global refinement parameters)
    cuda::refinement::RefinementParameters refinement;
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
    std::optional<BenchmarkParameters> _benchmarkParams;
};

}
