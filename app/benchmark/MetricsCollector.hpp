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
    std::vector<float> yPositions;   // For vertical profiles: Y positions
    std::vector<float> xVelocities;  // For vertical profiles: X velocities (U component)
    // For horizontal profiles: Y velocities (V component) stored here
    float timeStamp;
    float xPosition;  // For vertical: X position of profile line
    // For horizontal: Y position stored here
    std::string profileType = "vertical";  // "vertical" or "horizontal"
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

    // Basic metrics (always collected)
    float l2DensityErrorNorm = 0.F;
    float pressureFieldSmoothness = 0.F;
    float massConservationError = 0.F;
    float averageFrameTime = 0.F;
    float particleClusteringIndex = 0.F;

    // Basic time series (always collected)
    std::vector<float> frameTimes;
    std::vector<float> l2DensityErrors;

    // Enhanced metrics (collected for experiments with analytical solutions)
    std::vector<float> l2VelocityErrors;

    // Performance metrics (always collected)
    std::vector<float> cudaComputationTimes;
    std::vector<float> totalFrameTimes;
    std::vector<float> particlesPerSecond;
    std::vector<uint32_t> particleCounts;

    // Performance summary metrics
    float averageCudaTime = 0.0F;
    float averageThroughput = 0.0F;
    float cudaEfficiency = 0.0F;
    float peakThroughput = 0.0F;

    // Data collection (always collected)
    std::vector<VelocityProfile> velocityProfiles;
    std::vector<DensityProfile> densityProfiles;
    std::vector<ParticleSnapshot> particleSnapshots;

    // Simulation configuration
    struct SimulationConfig
    {
        // Common parameters
        float restDensity = 0.0F;
        float viscosityConstant = 0.0F;
        glm::vec3 domainMin;
        glm::vec3 domainMax;
        // Poiseuille-specific parameters
        float channelHeight = 0.0F;
        float channelLength = 0.0F;
        float channelWidth = 0.0F;
        float forceMagnitude = 0.0F;
        // Lid Driven Cavity parameters
        float cavitySize = 0.0F;
        float lidVelocity = 0.0F;

        // Taylor Green parameters
        float domainSize = 0.0F;
    } config;
};

class MetricsCollector
{
public:
    void reset();
    void initialize(const cuda::Simulation& simulation, float restDensity);
    void collectEnhancedMetrics(const cuda::Simulation& simulation,
                                float frameTime,
                                float cudaTime,
                                const BenchmarkResult::SimulationConfig& config,
                                cuda::Simulation::Parameters::TestCase experimentType);

    // Legacy method for backward compatibility
    void collectFrameMetrics(const cuda::Simulation& simulation, float frameTime);

    // Calculate results
    [[nodiscard]] auto calculateResults(cuda::Simulation::Parameters::TestCase experimentType,
                                        BenchmarkResult::SimulationType simulationType,
                                        const BenchmarkResult::SimulationConfig& config) const -> BenchmarkResult;

    static void saveToFile(const BenchmarkResult& result, const std::string& outputPath);

private:
    [[nodiscard]] auto calculateL2DensityErrorNorm() const -> float;
    [[nodiscard]] static auto calculatePressureFieldSmoothness() -> float;
    [[nodiscard]] static auto calculateParticleClusteringIndex() -> float;
    [[nodiscard]] auto calculateAverageFrameTime() const -> float;

    // Enhanced metrics calculation
    [[nodiscard]] auto calculateL2VelocityErrorNorm(const BenchmarkResult::SimulationConfig& config,
                                                    cuda::Simulation::Parameters::TestCase experimentType) const
        -> float;

    // Profile extraction methods
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

    [[nodiscard]] auto extractDensityProfile(const cuda::Simulation& simulation,
                                             float xPosition,
                                             float timeStamp,
                                             const BenchmarkResult::SimulationConfig& config) const -> DensityProfile;

    // Analytical solutions
    [[nodiscard]] static auto getAnalyticalVelocity(const glm::vec3& position,
                                                    float time,
                                                    const BenchmarkResult::SimulationConfig& config,
                                                    cuda::Simulation::Parameters::TestCase experimentType) -> glm::vec3;

    [[nodiscard]] static auto analyticalPoiseuilleVelocityX(float y, const BenchmarkResult::SimulationConfig& config)
        -> float;
    [[nodiscard]] static auto analyticalTaylorGreenVelocity(const glm::vec3& position,
                                                            float time,
                                                            const BenchmarkResult::SimulationConfig& config)
        -> glm::vec3;

    // Helper methods
    [[nodiscard]] static auto isInAnalysisRegion(const glm::vec3& position,
                                                 const BenchmarkResult::SimulationConfig& config,
                                                 cuda::Simulation::Parameters::TestCase experimentType) -> bool;

    // Data storage
    float _initialTotalMass {};
    float _restDensity {};
    std::vector<float> _frameTimes;
    std::vector<float> _l2DensityErrors;

    // Enhanced data storage
    std::vector<float> _l2VelocityErrors;
    std::vector<VelocityProfile> _velocityProfiles;
    std::vector<DensityProfile> _densityProfiles;
    std::vector<ParticleSnapshot> _particleSnapshots;

    // Performance metrics storage
    std::vector<float> _cudaComputationTimes;
    std::vector<float> _totalFrameTimes;
    std::vector<float> _particlesPerSecond;
    std::vector<uint32_t> _particleCounts;

    // Collection control
    uint32_t _profileCollectionInterval = 1000;   // Collect profiles every N frames
    uint32_t _snapshotCollectionInterval = 1000;  // Collect snapshots every N frames
    uint32_t _frameCounter = 0;

    // Velocity samples for L2 error calculation
    mutable std::vector<std::pair<glm::vec3, glm::vec3>> _velocitySamples;  // position, numerical_velocity
};

}
