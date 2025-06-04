#pragma once
#include <cuda/Simulation.cuh>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace sph::benchmark
{

struct VelocityProfile
{
    std::vector<float> yPositions;
    std::vector<float> xVelocities;
    float timeStamp;
    float xPosition;  // Which x-section this profile represents
    std::string profileType;
};

struct DensityProfile
{
    std::vector<float> yPositions;
    std::vector<float> densities;
    float timeStamp;
    float xPosition;
};

struct ParticleSnapshot
{
    std::vector<glm::vec4> positions;
    std::vector<glm::vec4> velocities;
    float timeStamp;
    uint32_t particleCount;
};

struct BenchmarkResult
{
    enum class SimulationType
    {
        Coarse,
        Fine,
        Adaptive
    };
    cuda::Simulation::Parameters::TestCase experimentType;
    SimulationType simulationType;
    // Existing metrics
    float l2DensityErrorNorm = 0.F;
    float pressureFieldSmoothness = 0.F;
    float massConservationError = 0.F;
    float averageFrameTime = 0.F;
    float particleClusteringIndex = 0.F;
    // Existing time series
    std::vector<float> frameTimes;
    std::vector<float> l2DensityErrors;
    std::vector<float> totalMasses;
    // NEW: Extended metrics for Poiseuille flow analysis
    std::vector<float> l2VelocityErrors;         // L2 norm of velocity error over time
    std::vector<float> l2DensityErrorsDetailed;  // More detailed density errors
                                                 // NEW: Performance metrics
    std::vector<float> cudaComputationTimes;     // Pure CUDA computation time [ms]
    std::vector<float> totalFrameTimes;          // Total frame time including overhead [ms]
    std::vector<float> particlesPerSecond;       // Computational throughput
    std::vector<uint32_t> particleCounts;        // Number of particles per frame
                                                 // NEW: Performance summary metrics
    float averageCudaTime = 0.0F;                // Average CUDA computation time [ms]
    float averageThroughput = 0.0F;              // Average particles per second
    float cudaEfficiency = 0.0F;                 // CUDA time / total time ratio
    float peakThroughput = 0.0F;                 // Peak computational throughput
                                                 // NEW: Velocity profiles collected during simulation
    std::vector<VelocityProfile> velocityProfiles;

    // NEW: Density profiles collected during simulation
    std::vector<DensityProfile> densityProfiles;

    std::vector<ParticleSnapshot> particleSnapshots;

    // NEW: Simulation parameters for analytical comparison
    struct SimulationConfig
    {
        // Poiseuille-specific parameters
        float channelHeight = 0.0F;
        float channelLength = 0.0F;
        float channelWidth = 0.0F;
        float forceMagnitude = 0.0F;
        // NEW: Lid Driven Cavity parameters
        float cavitySize = 0.0F;
        float lidVelocity = 0.0F;
        // Common parameters
        float restDensity = 0.0F;
        float viscosityConstant = 0.0F;
        glm::vec3 domainMin;
        glm::vec3 domainMax;
    } config;
};

class MetricsCollector
{
public:
    void reset();
    void initialize(const cuda::Simulation& simulation, float restDensity);
    void collectFrameMetrics(const cuda::Simulation& simulation, float frameTime);
    // Enhanced collection for Poiseuille flow with performance metrics
    void collectPoiseuilleMetrics(const cuda::Simulation& simulation,
                                  float frameTime,
                                  float cudaTime,
                                  const BenchmarkResult::SimulationConfig& config);
    // NEW: Enhanced collection for Lid Driven Cavity with performance metrics
    void collectCavityMetrics(const cuda::Simulation& simulation,
                              float frameTime,
                              float cudaTime,
                              const BenchmarkResult::SimulationConfig& config);
    [[nodiscard]] auto calculateResults(cuda::Simulation::Parameters::TestCase experimentType,
                                        BenchmarkResult::SimulationType simulationType) const -> BenchmarkResult;
    // Calculate results with configuration
    [[nodiscard]] auto calculateResults(cuda::Simulation::Parameters::TestCase experimentType,
                                        BenchmarkResult::SimulationType simulationType,
                                        const BenchmarkResult::SimulationConfig& config) const -> BenchmarkResult;
    static void saveToFile(const BenchmarkResult& result, const std::string& outputPath);

private:
    // Existing methods
    [[nodiscard]] auto calculateL2DensityErrorNorm() const -> float;
    [[nodiscard]] static auto calculatePressureFieldSmoothness() -> float;
    [[nodiscard]] auto calculateMassConservationError() const -> float;
    [[nodiscard]] static auto calculateParticleClusteringIndex() -> float;
    [[nodiscard]] auto calculateAverageFrameTime() const -> float;
    // Poiseuille-specific analysis methods
    [[nodiscard]] auto calculateL2VelocityErrorNorm(const BenchmarkResult::SimulationConfig& config) const -> float;
    [[nodiscard]] auto extractVelocityProfile(const cuda::Simulation& simulation,
                                              float xPosition,
                                              float timeStamp,
                                              const BenchmarkResult::SimulationConfig& config) const -> VelocityProfile;
    [[nodiscard]] auto extractDensityProfile(const cuda::Simulation& simulation,
                                             float xPosition,
                                             float timeStamp,
                                             const BenchmarkResult::SimulationConfig& config) const -> DensityProfile;
    // NEW: Lid Driven Cavity specific profile extraction
    [[nodiscard]] auto extractVelocityProfileVertical(const cuda::Simulation& simulation,
                                                      float xPosition,
                                                      float timeStamp,
                                                      const BenchmarkResult::SimulationConfig& config) const
        -> VelocityProfile;
    [[nodiscard]] auto extractVelocityProfileHorizontal(const cuda::Simulation& simulation,
                                                        float yPosition,
                                                        float timeStamp,
                                                        const BenchmarkResult::SimulationConfig& config) const
        -> VelocityProfile;
    // Performance metrics calculation
    [[nodiscard]] auto calculatePerformanceMetrics() const -> void;

    // Analytical solutions for comparison (Poiseuille only)
    [[nodiscard]] static auto analyticalVelocityX(float y, const BenchmarkResult::SimulationConfig& config) -> float;
    [[nodiscard]] static auto analyticalDensity(const BenchmarkResult::SimulationConfig& config) -> float;

    // Check if position is in middle 10% of channel (Poiseuille specific)
    [[nodiscard]] static auto isInMiddleRegion(const glm::vec3& position,
                                               const BenchmarkResult::SimulationConfig& config) -> bool;

    // NEW: Check if position is in central region of cavity
    [[nodiscard]] static auto isInCentralRegion(const glm::vec3& position,
                                                const BenchmarkResult::SimulationConfig& config) -> bool;

    // Existing data
    float _initialTotalMass {};
    float _restDensity {};
    std::vector<float> _frameTimes;
    std::vector<float> _l2DensityErrors;
    std::vector<float> _totalMasses;
    // Enhanced data storage
    std::vector<float> _l2VelocityErrors;
    std::vector<VelocityProfile> _velocityProfiles;
    std::vector<DensityProfile> _densityProfiles;
    // Performance metrics storage
    std::vector<float> _cudaComputationTimes;
    std::vector<float> _totalFrameTimes;
    std::vector<float> _particlesPerSecond;
    std::vector<uint32_t> _particleCounts;

    // Profile collection control
    uint32_t _profileCollectionInterval = 100;  // Collect profiles every N frames
    uint32_t _frameCounter = 0;

    // Analytical velocity samples for L2 error calculation (Poiseuille specific)
    mutable std::vector<std::pair<glm::vec3, float>> _velocitySamples;  // position, numerical_vx

    std::vector<ParticleSnapshot> _particleSnapshots;
    uint32_t _snapshotCollectionInterval = 100;  // Every 500 frames
};

}
