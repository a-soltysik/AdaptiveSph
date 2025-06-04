#include "MetricsCollector.hpp"

#include <fmt/format.h>
#include <panda/Logger.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <numeric>
#include <string>
#include <utility>

#include "cuda/Simulation.cuh"

namespace sph::benchmark
{

void MetricsCollector::reset()
{
    _initialTotalMass = 0.F;
    _restDensity = 0.F;
    _frameTimes.clear();
    _l2DensityErrors.clear();
    _totalMasses.clear();

    // Extended data
    _l2VelocityErrors.clear();
    _velocityProfiles.clear();
    _densityProfiles.clear();
    _velocitySamples.clear();
    _frameCounter = 0;

    // Performance data
    _cudaComputationTimes.clear();
    _totalFrameTimes.clear();
    _particlesPerSecond.clear();
    _particleCounts.clear();

    // NEW: Reset particle snapshots
    _particleSnapshots.clear();
}

void MetricsCollector::initialize(const cuda::Simulation& simulation, float restDensity)
{
    _restDensity = restDensity;

    auto particleCount = simulation.getParticlesCount();
    if (particleCount > 0)
    {
        _initialTotalMass = static_cast<float>(particleCount) * restDensity;
    }
}

void MetricsCollector::collectFrameMetrics(const cuda::Simulation& simulation, float frameTime)
{
    _frameTimes.push_back(frameTime);

    auto densityDeviations = simulation.updateDensityDeviations();
    auto particleCount = simulation.getParticlesCount();

    if (particleCount > 0)
    {
        auto sumErrorSquared = 0.F;
        for (uint32_t i = 0; i < particleCount; ++i)
        {
            const auto error = densityDeviations[i];
            sumErrorSquared += error * error;
        }
        const auto l2Norm = std::sqrt(sumErrorSquared / static_cast<float>(particleCount));
        _l2DensityErrors.push_back(l2Norm);

        const auto currentTotalMass = static_cast<float>(particleCount) * _restDensity;
        _totalMasses.push_back(currentTotalMass);
    }
}

void MetricsCollector::collectPoiseuilleMetrics(const cuda::Simulation& simulation,
                                                float frameTime,
                                                float cudaTime,
                                                const BenchmarkResult::SimulationConfig& config)
{
    // Collect standard metrics
    collectFrameMetrics(simulation, frameTime);

    _frameCounter++;

    auto particleCount = simulation.getParticlesCount();
    if (particleCount == 0)
    {
        return;
    }

    // NEW: Collect performance metrics
    _cudaComputationTimes.push_back(cudaTime);
    _totalFrameTimes.push_back(frameTime);
    _particleCounts.push_back(particleCount);

    // Calculate throughput (particles per second)
    if (cudaTime > 0.0F)
    {
        const auto throughput = static_cast<float>(particleCount) / (cudaTime / 1000.0F);  // Convert ms to seconds
        _particlesPerSecond.push_back(throughput);
    }
    else
    {
        _particlesPerSecond.push_back(0.0F);
    }

    // Collect velocity samples from middle region for L2 error calculation
    _velocitySamples.clear();

    // Get particle data for velocity error calculation
    const auto particleData = simulation.getParticleDataSnapshot();

    // Sample velocities from particles in the middle region
    for (uint32_t i = 0; i < particleData.particleCount; ++i)
    {
        const auto& position = particleData.positions[i];
        if (isInMiddleRegion(glm::vec3(position), config))
        {
            const auto& velocity = particleData.velocities[i];
            _velocitySamples.emplace_back(glm::vec3(position), velocity.x);
        }
    }

    // Calculate current L2 velocity error
    const auto currentL2VelError = calculateL2VelocityErrorNorm(config);
    _l2VelocityErrors.push_back(currentL2VelError);

    // Collect velocity and density profiles at specified intervals
    if (_frameCounter % _profileCollectionInterval == 0)
    {
        const auto currentTime = static_cast<float>(_frameCounter) * frameTime;

        // Collect profiles at middle x-position
        const auto middleX = (config.domainMin.x + config.domainMax.x) * 0.5F;

        auto velProfile = extractVelocityProfile(simulation, middleX, currentTime, config);
        auto densProfile = extractDensityProfile(simulation, middleX, currentTime, config);

        _velocityProfiles.push_back(std::move(velProfile));
        _densityProfiles.push_back(std::move(densProfile));
    }
}

auto MetricsCollector::extractVelocityProfile(const cuda::Simulation& simulation,
                                              float xPosition,
                                              float timeStamp,
                                              const BenchmarkResult::SimulationConfig& config) const -> VelocityProfile
{
    VelocityProfile profile;
    profile.xPosition = xPosition;
    profile.timeStamp = timeStamp;

    // Get particle data from simulation
    const auto particleData = simulation.getParticleDataSnapshot();

    // Collect particles near the specified x-position
    const auto xTolerance = 0.1F;                    // Tolerance for x-position matching
    std::vector<std::pair<float, float>> yVelPairs;  // y-position, x-velocity pairs

    for (uint32_t i = 0; i < particleData.particleCount; ++i)
    {
        const auto& position = particleData.positions[i];
        const auto& velocity = particleData.velocities[i];

        // Check if particle is near the desired x-position
        if (std::abs(position.x - xPosition) <= xTolerance)
        {
            yVelPairs.emplace_back(position.y, velocity.x);
        }
    }

    // Sort by y-position for consistent profiling
    std::sort(yVelPairs.begin(), yVelPairs.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // Extract sorted data
    profile.yPositions.reserve(yVelPairs.size());
    profile.xVelocities.reserve(yVelPairs.size());

    for (const auto& [y, vx] : yVelPairs)
    {
        profile.yPositions.push_back(y);
        profile.xVelocities.push_back(vx);
    }

    panda::log::Info("Extracted velocity profile at x={:.3f}, t={:.3f} with {} particles",
                     xPosition,
                     timeStamp,
                     profile.yPositions.size());

    return profile;
}

auto MetricsCollector::extractDensityProfile(const cuda::Simulation& simulation,
                                             float xPosition,
                                             float timeStamp,
                                             const BenchmarkResult::SimulationConfig& config) const -> DensityProfile
{
    DensityProfile profile;
    profile.xPosition = xPosition;
    profile.timeStamp = timeStamp;

    // Get particle data from simulation
    const auto particleData = simulation.getParticleDataSnapshot();

    // Collect particles near the specified x-position
    const auto xTolerance = 0.1F;                     // Tolerance for x-position matching
    std::vector<std::pair<float, float>> yDensPairs;  // y-position, density pairs

    for (uint32_t i = 0; i < particleData.particleCount; ++i)
    {
        const auto& position = particleData.positions[i];
        const auto density = particleData.densities[i];

        // Check if particle is near the desired x-position
        if (std::abs(position.x - xPosition) <= xTolerance)
        {
            yDensPairs.emplace_back(position.y, density);
        }
    }

    // Sort by y-position for consistent profiling
    std::sort(yDensPairs.begin(), yDensPairs.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // Extract sorted data
    profile.yPositions.reserve(yDensPairs.size());
    profile.densities.reserve(yDensPairs.size());

    for (const auto& [y, density] : yDensPairs)
    {
        profile.yPositions.push_back(y);
        profile.densities.push_back(density);
    }

    panda::log::Info("Extracted density profile at x={:.3f}, t={:.3f} with {} particles",
                     xPosition,
                     timeStamp,
                     profile.yPositions.size());

    return profile;
}

auto MetricsCollector::analyticalVelocityX(float y, const BenchmarkResult::SimulationConfig& config) -> float
{
    // Poiseuille flow analytical solution:
    // vₓ(y) = (ρ * g * h²) / (8 * μ) * (1 - (2y/h)²)
    // where μ = viscosityConstant * restDensity

    const auto rho = config.restDensity;
    const auto g = config.forceMagnitude;
    const auto h = config.channelHeight;
    const auto mu = config.viscosityConstant * rho;

    const auto coefficient = (rho * g * h * h) / (8.0F * mu);
    const auto yNormalized = 2.0F * y / h;  // y ∈ [-1, 1]

    return coefficient * (1.0F - yNormalized * yNormalized);
}

auto MetricsCollector::analyticalDensity(const BenchmarkResult::SimulationConfig& config) -> float
{
    // For incompressible Poiseuille flow, density should be constant everywhere
    return config.restDensity;
}

auto MetricsCollector::isInMiddleRegion(const glm::vec3& position, const BenchmarkResult::SimulationConfig& config)
    -> bool
{
    // Check if position is in middle 10% of channel length (x-direction)
    const auto domainLength = config.domainMax.x - config.domainMin.x;
    const auto middleStart = config.domainMin.x + 0.45F * domainLength;
    const auto middleEnd = config.domainMin.x + 0.55F * domainLength;

    return position.x >= middleStart && position.x <= middleEnd;
}

auto MetricsCollector::calculateL2VelocityErrorNorm(const BenchmarkResult::SimulationConfig& config) const -> float
{
    if (_velocitySamples.empty())
    {
        return 0.0F;
    }

    auto sumErrorSquared = 0.0F;
    auto validSamples = 0;

    for (const auto& [position, numericalVx] : _velocitySamples)
    {
        if (isInMiddleRegion(position, config))
        {
            const auto analyticalVx = analyticalVelocityX(position.y, config);
            const auto error = numericalVx - analyticalVx;
            sumErrorSquared += error * error;
            validSamples++;
        }
    }

    if (validSamples > 0)
    {
        return std::sqrt(sumErrorSquared / static_cast<float>(validSamples));
    }

    return 0.0F;
}

auto MetricsCollector::calculateResults(cuda::Simulation::Parameters::TestCase experimentType,
                                        BenchmarkResult::SimulationType simulationType) const -> BenchmarkResult
{
    BenchmarkResult result;
    result.experimentType = experimentType;
    result.simulationType = simulationType;

    result.l2DensityErrorNorm = calculateL2DensityErrorNorm();
    result.pressureFieldSmoothness = calculatePressureFieldSmoothness();
    result.massConservationError = calculateMassConservationError();
    result.averageFrameTime = calculateAverageFrameTime();
    result.particleClusteringIndex = calculateParticleClusteringIndex();

    result.frameTimes = _frameTimes;
    result.l2DensityErrors = _l2DensityErrors;
    result.totalMasses = _totalMasses;

    return result;
}

auto MetricsCollector::calculateResults(cuda::Simulation::Parameters::TestCase experimentType,
                                        BenchmarkResult::SimulationType simulationType,
                                        const BenchmarkResult::SimulationConfig& config) const -> BenchmarkResult
{
    auto result = calculateResults(experimentType, simulationType);

    // Add extended results
    result.config = config;
    result.l2VelocityErrors = _l2VelocityErrors;
    result.velocityProfiles = _velocityProfiles;
    result.densityProfiles = _densityProfiles;

    // NEW: Add particle snapshots
    result.particleSnapshots = _particleSnapshots;

    // Performance metrics
    result.cudaComputationTimes = _cudaComputationTimes;
    result.totalFrameTimes = _totalFrameTimes;
    result.particlesPerSecond = _particlesPerSecond;
    result.particleCounts = _particleCounts;

    // Calculate performance summary metrics
    if (!_cudaComputationTimes.empty())
    {
        result.averageCudaTime = std::accumulate(_cudaComputationTimes.begin(), _cudaComputationTimes.end(), 0.0F) /
                                 static_cast<float>(_cudaComputationTimes.size());
    }

    if (!_particlesPerSecond.empty())
    {
        result.averageThroughput = std::accumulate(_particlesPerSecond.begin(), _particlesPerSecond.end(), 0.0F) /
                                   static_cast<float>(_particlesPerSecond.size());
        result.peakThroughput = *std::max_element(_particlesPerSecond.begin(), _particlesPerSecond.end());
    }

    if (!_totalFrameTimes.empty() && !_cudaComputationTimes.empty())
    {
        const auto avgTotalTime = std::accumulate(_totalFrameTimes.begin(), _totalFrameTimes.end(), 0.0F) /
                                  static_cast<float>(_totalFrameTimes.size());
        if (avgTotalTime > 0.0F)
        {
            result.cudaEfficiency = result.averageCudaTime / avgTotalTime;
        }
    }

    return result;
}

// Update this part of MetricsCollector::saveToFile() method

void MetricsCollector::saveToFile(const BenchmarkResult& result, const std::string& outputPath)
{
    using json = nlohmann::json;

    json resultJson;
    resultJson["experimentType"] = static_cast<uint32_t>(result.experimentType);
    resultJson["simulationType"] = static_cast<uint32_t>(result.simulationType);

    // Existing metrics
    resultJson["metrics"]["l2DensityErrorNorm"] = result.l2DensityErrorNorm;
    resultJson["metrics"]["pressureFieldSmoothness"] = result.pressureFieldSmoothness;
    resultJson["metrics"]["massConservationError"] = result.massConservationError;
    resultJson["metrics"]["averageFrameTime"] = result.averageFrameTime;
    resultJson["metrics"]["particleClusteringIndex"] = result.particleClusteringIndex;

    // Existing time series
    resultJson["timeSeries"]["frameTimes"] = result.frameTimes;
    resultJson["timeSeries"]["l2DensityErrors"] = result.l2DensityErrors;
    resultJson["timeSeries"]["totalMasses"] = result.totalMasses;

    // Extended metrics for enhanced experiments
    if (!result.l2VelocityErrors.empty())
    {
        resultJson["timeSeries"]["l2VelocityErrors"] = result.l2VelocityErrors;
    }

    // Performance metrics
    if (!result.cudaComputationTimes.empty())
    {
        resultJson["performance"]["cudaComputationTimes"] = result.cudaComputationTimes;
        resultJson["performance"]["totalFrameTimes"] = result.totalFrameTimes;
        resultJson["performance"]["particlesPerSecond"] = result.particlesPerSecond;
        resultJson["performance"]["particleCounts"] = result.particleCounts;

        // Performance summary
        resultJson["performance"]["summary"]["averageCudaTime"] = result.averageCudaTime;
        resultJson["performance"]["summary"]["averageThroughput"] = result.averageThroughput;
        resultJson["performance"]["summary"]["peakThroughput"] = result.peakThroughput;
        resultJson["performance"]["summary"]["cudaEfficiency"] = result.cudaEfficiency;
    }

    // Velocity profiles
    if (!result.velocityProfiles.empty())
    {
        json velocityProfilesJson = json::array();
        for (const auto& profile : result.velocityProfiles)
        {
            json profileJson;
            profileJson["timeStamp"] = profile.timeStamp;
            profileJson["xPosition"] = profile.xPosition;
            profileJson["yPositions"] = profile.yPositions;
            profileJson["xVelocities"] = profile.xVelocities;
            profileJson["profileType"] = profile.profileType;  // NEW: Profile type
            velocityProfilesJson.push_back(profileJson);
        }
        resultJson["profiles"]["velocity"] = velocityProfilesJson;
    }

    // Density profiles
    if (!result.densityProfiles.empty())
    {
        json densityProfilesJson = json::array();
        for (const auto& profile : result.densityProfiles)
        {
            json profileJson;
            profileJson["timeStamp"] = profile.timeStamp;
            profileJson["xPosition"] = profile.xPosition;
            profileJson["yPositions"] = profile.yPositions;
            profileJson["densities"] = profile.densities;
            densityProfilesJson.push_back(profileJson);
        }
        resultJson["profiles"]["density"] = densityProfilesJson;
    }

    if (!result.particleSnapshots.empty())
    {
        json snapshotsJson = json::array();
        for (const auto& snapshot : result.particleSnapshots)
        {
            json snapshotJson;
            snapshotJson["timeStamp"] = snapshot.timeStamp;
            snapshotJson["particleCount"] = snapshot.particleCount;
            // Convert positions to JSON array
            json positionsJson = json::array();
            for (const auto& pos : snapshot.positions)
            {
                positionsJson.push_back({pos.x, pos.y, pos.z});
            }
            snapshotJson["positions"] = positionsJson;
            // Convert velocities to JSON array
            json velocitiesJson = json::array();
            for (const auto& vel : snapshot.velocities)
            {
                velocitiesJson.push_back({vel.x, vel.y, vel.z});
            }
            snapshotJson["velocities"] = velocitiesJson;

            snapshotsJson.push_back(snapshotJson);
        }
        resultJson["particleSnapshots"] = snapshotsJson;

        panda::log::Info("Saved {} particle snapshots to JSON", result.particleSnapshots.size());
    }

    // Simulation configuration for analytical comparison
    json configJson;
    // Poiseuille-specific parameters
    configJson["channelHeight"] = result.config.channelHeight;
    configJson["channelLength"] = result.config.channelLength;
    configJson["channelWidth"] = result.config.channelWidth;
    configJson["forceMagnitude"] = result.config.forceMagnitude;
    // NEW: Lid Driven Cavity specific parameters
    configJson["cavitySize"] = result.config.cavitySize;
    configJson["lidVelocity"] = result.config.lidVelocity;
    // Common parameters
    configJson["restDensity"] = result.config.restDensity;
    configJson["viscosityConstant"] = result.config.viscosityConstant;
    configJson["domainMin"] = {result.config.domainMin.x, result.config.domainMin.y, result.config.domainMin.z};
    configJson["domainMax"] = {result.config.domainMax.x, result.config.domainMax.y, result.config.domainMax.z};
    resultJson["config"] = configJson;

    const auto filename = fmt::format("{}/{}_{}.json",
                                      outputPath,
                                      static_cast<uint32_t>(result.experimentType),
                                      static_cast<uint32_t>(result.simulationType));

    std::ofstream file(filename);
    if (file.is_open())
    {
        file << resultJson.dump(4);
        file.close();
        panda::log::Info("Saved enhanced benchmark results to: {}", filename);
    }
    else
    {
        panda::log::Error("Failed to save benchmark results to: {}", filename);
    }
}

// Existing method implementations remain the same...
auto MetricsCollector::calculateL2DensityErrorNorm() const -> float
{
    if (_l2DensityErrors.empty())
    {
        return 0.0F;
    }

    return std::accumulate(_l2DensityErrors.begin(), _l2DensityErrors.end(), 0.0F) /
           static_cast<float>(_l2DensityErrors.size());
}

auto MetricsCollector::calculatePressureFieldSmoothness() -> float
{
    return 0.F;
}

auto MetricsCollector::calculateMassConservationError() const -> float
{
    if (_totalMasses.empty() || _initialTotalMass <= 0.0F)
    {
        return 0.0F;
    }

    const auto finalTotalMass = _totalMasses.back();
    return std::abs(finalTotalMass - _initialTotalMass) / _initialTotalMass;
}

auto MetricsCollector::calculateParticleClusteringIndex() -> float
{
    return 0.0F;
}

auto MetricsCollector::calculateAverageFrameTime() const -> float
{
    if (_frameTimes.empty())
    {
        return 0.0F;
    }

    return std::accumulate(_frameTimes.begin(), _frameTimes.end(), 0.0F) / static_cast<float>(_frameTimes.size());
}

void MetricsCollector::collectCavityMetrics(const cuda::Simulation& simulation,
                                            float frameTime,
                                            float cudaTime,
                                            const BenchmarkResult::SimulationConfig& config)
{
    // Collect standard metrics
    collectFrameMetrics(simulation, frameTime);

    _frameCounter++;

    auto particleCount = simulation.getParticlesCount();
    if (particleCount == 0)
    {
        return;
    }

    // Collect performance metrics
    _cudaComputationTimes.push_back(cudaTime);
    _totalFrameTimes.push_back(frameTime);
    _particleCounts.push_back(particleCount);

    // Calculate throughput (particles per second)
    if (cudaTime > 0.0F)
    {
        const auto throughput = static_cast<float>(particleCount) / (cudaTime / 1000.0F);
        _particlesPerSecond.push_back(throughput);
    }
    else
    {
        _particlesPerSecond.push_back(0.0F);
    }

    // NEW: Collect particle snapshots for velocity vector plots
    if (_frameCounter % _snapshotCollectionInterval == 0)
    {
        const auto currentTime = static_cast<float>(_frameCounter) * frameTime;
        ParticleSnapshot snapshot;
        snapshot.timeStamp = currentTime;
        snapshot.particleCount = particleCount;
        // Get full particle data
        auto particleData = simulation.getParticleDataSnapshot();
        snapshot.positions = std::move(particleData.positions);
        snapshot.velocities = std::move(particleData.velocities);
        _particleSnapshots.push_back(std::move(snapshot));
        panda::log::Info("Collected particle snapshot at frame {} (t={:.3f}) with {} particles",
                         _frameCounter,
                         currentTime,
                         particleCount);
    }

    // Collect velocity and density profiles at specified intervals
    if (_frameCounter % _profileCollectionInterval == 0)
    {
        const auto currentTime = static_cast<float>(_frameCounter) * frameTime;

        // Collect profiles at cavity center lines
        const auto centerX = (config.domainMin.x + config.domainMax.x) * 0.5F;
        const auto centerY = (config.domainMin.y + config.domainMax.y) * 0.5F;

        // Vertical profile: U velocity along y-axis at center x
        auto verticalVelProfile = extractVelocityProfileVertical(simulation, centerX, currentTime, config);
        verticalVelProfile.profileType = "vertical";  // NEW: Mark profile type
                                                      // Horizontal profile: V velocity along x-axis at center y
        auto horizontalVelProfile = extractVelocityProfileHorizontal(simulation, centerY, currentTime, config);
        horizontalVelProfile.profileType = "horizontal";  // NEW: Mark profile type

        _velocityProfiles.push_back(std::move(verticalVelProfile));
        _velocityProfiles.push_back(std::move(horizontalVelProfile));

        // Also collect density profile at center x
        auto densProfile = extractDensityProfile(simulation, centerX, currentTime, config);
        _densityProfiles.push_back(std::move(densProfile));
    }
}

auto MetricsCollector::extractVelocityProfileVertical(const cuda::Simulation& simulation,
                                                      float xPosition,
                                                      float timeStamp,
                                                      const BenchmarkResult::SimulationConfig& config) const
    -> VelocityProfile
{
    VelocityProfile profile;
    profile.xPosition = xPosition;
    profile.timeStamp = timeStamp;

    // Get particle data from simulation
    const auto particleData = simulation.getParticleDataSnapshot();

    // Collect particles near the specified x-position
    const auto xTolerance = 0.1F;                    // Tolerance for x-position matching
    std::vector<std::pair<float, float>> yVelPairs;  // y-position, x-velocity pairs (U component)

    for (uint32_t i = 0; i < particleData.particleCount; ++i)
    {
        const auto& position = particleData.positions[i];
        const auto& velocity = particleData.velocities[i];

        // Check if particle is near the desired x-position
        if (std::abs(position.x - xPosition) <= xTolerance)
        {
            yVelPairs.emplace_back(position.y, velocity.x);  // U velocity component
        }
    }

    // Sort by y-position for consistent profiling
    std::sort(yVelPairs.begin(), yVelPairs.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // Extract sorted data
    profile.yPositions.reserve(yVelPairs.size());
    profile.xVelocities.reserve(yVelPairs.size());

    for (const auto& [y, vx] : yVelPairs)
    {
        profile.yPositions.push_back(y);
        profile.xVelocities.push_back(vx);
    }

    panda::log::Info("Extracted vertical velocity profile (U vs Y) at x={:.3f}, t={:.3f} with {} particles",
                     xPosition,
                     timeStamp,
                     profile.yPositions.size());

    return profile;
}

auto MetricsCollector::extractVelocityProfileHorizontal(const cuda::Simulation& simulation,
                                                        float yPosition,
                                                        float timeStamp,
                                                        const BenchmarkResult::SimulationConfig& config) const
    -> VelocityProfile
{
    VelocityProfile profile;
    profile.xPosition = yPosition;  // Store y-position in xPosition field for horizontal profile
    profile.timeStamp = timeStamp;

    // Get particle data from simulation
    const auto particleData = simulation.getParticleDataSnapshot();

    // Collect particles near the specified y-position
    const auto yTolerance = 0.1F;                    // Tolerance for y-position matching
    std::vector<std::pair<float, float>> xVelPairs;  // x-position, y-velocity pairs (V component)

    for (uint32_t i = 0; i < particleData.particleCount; ++i)
    {
        const auto& position = particleData.positions[i];
        const auto& velocity = particleData.velocities[i];

        // Check if particle is near the desired y-position
        if (std::abs(position.y - yPosition) <= yTolerance)
        {
            xVelPairs.emplace_back(position.x, velocity.y);  // V velocity component
        }
    }

    // Sort by x-position for consistent profiling
    std::sort(xVelPairs.begin(), xVelPairs.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // Extract sorted data - reusing existing structure
    profile.yPositions.reserve(xVelPairs.size());   // Actually stores x-positions for horizontal profile
    profile.xVelocities.reserve(xVelPairs.size());  // Actually stores y-velocities for horizontal profile

    for (const auto& [x, vy] : xVelPairs)
    {
        profile.yPositions.push_back(x);    // x-positions in yPositions field
        profile.xVelocities.push_back(vy);  // V velocity in xVelocities field
    }

    panda::log::Info("Extracted horizontal velocity profile (V vs X) at y={:.3f}, t={:.3f} with {} particles",
                     yPosition,
                     timeStamp,
                     profile.yPositions.size());

    return profile;
}

auto MetricsCollector::isInCentralRegion(const glm::vec3& position, const BenchmarkResult::SimulationConfig& config)
    -> bool
{
    // Check if position is in central 50% of cavity (25% margin from each wall)
    const auto domainSizeX = config.domainMax.x - config.domainMin.x;
    const auto domainSizeY = config.domainMax.y - config.domainMin.y;
    const auto centralMargin = 0.25F;  // 25% margin from walls

    const auto centralStartX = config.domainMin.x + centralMargin * domainSizeX;
    const auto centralEndX = config.domainMax.x - centralMargin * domainSizeX;

    const auto centralStartY = config.domainMin.y + centralMargin * domainSizeY;
    const auto centralEndY = config.domainMax.y - centralMargin * domainSizeY;

    return position.x >= centralStartX && position.x <= centralEndX && position.y >= centralStartY &&
           position.y <= centralEndY;
}

}
