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
#include "glm/ext/scalar_constants.hpp"
#include "glm/geometric.hpp"

namespace sph::benchmark
{

void MetricsCollector::reset()
{
    _initialTotalMass = 0.F;
    _restDensity = 0.F;
    _frameTimes.clear();
    _l2DensityErrors.clear();
    // Enhanced data
    _l2VelocityErrors.clear();
    _velocityProfiles.clear();
    _densityProfiles.clear();
    _particleSnapshots.clear();
    _frameCounter = 0;

    // Performance data
    _cudaComputationTimes.clear();
    _totalFrameTimes.clear();
    _particlesPerSecond.clear();
    _particleCounts.clear();

    // Velocity samples
    _velocitySamples.clear();
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

void MetricsCollector::collectEnhancedMetrics(const cuda::Simulation& simulation,
                                              float frameTime,
                                              float cudaTime,
                                              const BenchmarkResult::SimulationConfig& config,
                                              cuda::Simulation::Parameters::TestCase experimentType)
{
    // 1. Collect basic frame metrics
    collectFrameMetrics(simulation, frameTime);

    _frameCounter++;

    auto particleCount = simulation.getParticlesCount();
    if (particleCount == 0)
    {
        return;
    }

    // 2. Collect performance metrics
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

    // 3. Collect particle snapshots (for all experiments)
    if (_frameCounter % _snapshotCollectionInterval == 0)
    {
        const auto currentTime = static_cast<float>(_frameCounter) * frameTime;
        ParticleSnapshot snapshot;
        snapshot.timeStamp = currentTime;
        snapshot.particleCount = particleCount;

        auto particleData = simulation.getParticleDataSnapshot();
        snapshot.positions = std::move(particleData.positions);
        snapshot.velocities = std::move(particleData.velocities);

        _particleSnapshots.push_back(std::move(snapshot));

        panda::log::Info("Collected particle snapshot at frame {} (t={:.3f}) with {} particles",
                         _frameCounter,
                         currentTime,
                         particleCount);
    }

    // 4. Collect velocity samples for analytical comparison (where applicable)
    if (experimentType == cuda::Simulation::Parameters::TestCase::PoiseuilleFlow ||
        experimentType == cuda::Simulation::Parameters::TestCase::TaylorGreenVortex)
    {
        _velocitySamples.clear();
        const auto particleData = simulation.getParticleDataSnapshot();
        const auto currentTime = static_cast<float>(_frameCounter) * frameTime;

        for (uint32_t i = 0; i < particleData.particleCount; ++i)
        {
            const auto position = glm::vec3(particleData.positions[i]);
            const auto velocity = glm::vec3(particleData.velocities[i]);

            if (isInAnalysisRegion(position, config, experimentType))
            {
                _velocitySamples.emplace_back(position, velocity);
            }
        }

        // Calculate current L2 velocity error
        const auto currentL2VelError = calculateL2VelocityErrorNorm(config, experimentType);
        _l2VelocityErrors.push_back(currentL2VelError);
    }

    // 5. Collect velocity and density profiles at specified intervals
    if (_frameCounter % _profileCollectionInterval == 0)
    {
        const auto currentTime = static_cast<float>(_frameCounter) * frameTime;

        switch (experimentType)
        {
        case cuda::Simulation::Parameters::TestCase::PoiseuilleFlow:
        {
            // Collect vertical profile at middle x-position
            const auto middleX = (config.domainMin.x + config.domainMax.x) * 0.5F;
            auto velProfile = extractVelocityProfileVertical(simulation, middleX, currentTime, config);
            velProfile.profileType = "vertical";
            _velocityProfiles.push_back(std::move(velProfile));

            auto densProfile = extractDensityProfile(simulation, middleX, currentTime, config);
            _densityProfiles.push_back(std::move(densProfile));
        }
        break;

        case cuda::Simulation::Parameters::TestCase::LidDrivenCavity:
        {
            // Collect both vertical and horizontal profiles at cavity center
            const auto centerX = (config.domainMin.x + config.domainMax.x) * 0.5F;
            const auto centerY = (config.domainMin.y + config.domainMax.y) * 0.5F;

            // Vertical profile: U velocity along y-axis at center x
            auto verticalVelProfile = extractVelocityProfileVertical(simulation, centerX, currentTime, config);
            verticalVelProfile.profileType = "vertical";
            _velocityProfiles.push_back(std::move(verticalVelProfile));

            // Horizontal profile: V velocity along x-axis at center y
            auto horizontalVelProfile = extractVelocityProfileHorizontal(simulation, centerY, currentTime, config);
            horizontalVelProfile.profileType = "horizontal";
            _velocityProfiles.push_back(std::move(horizontalVelProfile));

            // Density profile at center x
            auto densProfile = extractDensityProfile(simulation, centerX, currentTime, config);
            _densityProfiles.push_back(std::move(densProfile));
        }
        break;

        case cuda::Simulation::Parameters::TestCase::TaylorGreenVortex:
        {
            // Collect vertical profile at middle x-position for Taylor-Green
            const auto middleX = (config.domainMin.x + config.domainMax.x) * 0.5F;
            auto velProfile = extractVelocityProfileVertical(simulation, middleX, currentTime, config);
            velProfile.profileType = "vertical";
            _velocityProfiles.push_back(std::move(velProfile));

            auto densProfile = extractDensityProfile(simulation, middleX, currentTime, config);
            _densityProfiles.push_back(std::move(densProfile));
        }
        break;

        default:
            // For other experiments, collect basic vertical profile
            {
                const auto middleX = (config.domainMin.x + config.domainMax.x) * 0.5F;
                auto velProfile = extractVelocityProfileVertical(simulation, middleX, currentTime, config);
                velProfile.profileType = "vertical";
                _velocityProfiles.push_back(std::move(velProfile));
            }
            break;
        }
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
    }
}

auto MetricsCollector::calculateResults(cuda::Simulation::Parameters::TestCase experimentType,
                                        BenchmarkResult::SimulationType simulationType,
                                        const BenchmarkResult::SimulationConfig& config) const -> BenchmarkResult
{
    BenchmarkResult result;
    result.experimentType = experimentType;
    result.simulationType = simulationType;
    result.config = config;

    // Basic metrics
    result.l2DensityErrorNorm = calculateL2DensityErrorNorm();
    result.pressureFieldSmoothness = calculatePressureFieldSmoothness();
    result.averageFrameTime = calculateAverageFrameTime();
    result.particleClusteringIndex = calculateParticleClusteringIndex();

    // Basic time series
    result.frameTimes = _frameTimes;
    result.l2DensityErrors = _l2DensityErrors;

    // Enhanced metrics
    result.l2VelocityErrors = _l2VelocityErrors;
    result.velocityProfiles = _velocityProfiles;
    result.densityProfiles = _densityProfiles;
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

auto MetricsCollector::getAnalyticalVelocity(const glm::vec3& position,
                                             float time,
                                             const BenchmarkResult::SimulationConfig& config,
                                             cuda::Simulation::Parameters::TestCase experimentType) -> glm::vec3
{
    switch (experimentType)
    {
    case cuda::Simulation::Parameters::TestCase::PoiseuilleFlow:
        return glm::vec3(analyticalPoiseuilleVelocityX(position.y, config), 0.0F, 0.0F);

    case cuda::Simulation::Parameters::TestCase::TaylorGreenVortex:
        return analyticalTaylorGreenVelocity(position, time, config);

    default:
        return glm::vec3(0.0F);
    }
}

auto MetricsCollector::analyticalPoiseuilleVelocityX(float y, const BenchmarkResult::SimulationConfig& config) -> float
{
    // Poiseuille flow analytical solution: vₓ(y) = (ρ * g * h²) / (8 * μ) * (1 - (2y/h)²)
    const auto rho = config.restDensity;
    const auto g = config.forceMagnitude;
    const auto h = config.channelHeight;
    const auto mu = config.viscosityConstant * rho;

    if (h <= 0.0F || mu <= 0.0F)
    {
        return 0.0F;
    }

    const auto coefficient = (rho * g * h * h) / (8.0F * mu);
    const auto yNormalized = 2.0F * y / h;  // y ∈ [-1, 1]

    return coefficient * (1.0F - yNormalized * yNormalized);
}

auto MetricsCollector::analyticalTaylorGreenVelocity(const glm::vec3& position,
                                                     float time,
                                                     const BenchmarkResult::SimulationConfig& config) -> glm::vec3
{
    // Taylor-Green vortex analytical solution with time decay
    // vₓ = cos(x) * sin(y) * cos(z) * exp(-2νt)
    // vᵧ = -sin(x) * cos(y) * cos(z) * exp(-2νt)
    // vᵤ = 0

    // Map position to [0, 2π] range
    const auto domainSize = config.domainMax - config.domainMin;
    const auto x = (position.x - config.domainMin.x) / domainSize.x * 2.0F * glm::pi<float>();
    const auto y = (position.y - config.domainMin.y) / domainSize.y * 2.0F * glm::pi<float>();
    const auto z = (position.z - config.domainMin.z) / domainSize.z * 2.0F * glm::pi<float>();

    // Time decay factor
    const auto decay = std::exp(-2.0F * config.viscosityConstant * time);

    const auto u = std::cos(x) * std::sin(y) * std::cos(z) * decay;
    const auto v = -std::sin(x) * std::cos(y) * std::cos(z) * decay;
    const auto w = 0.0F;

    return glm::vec3(u, v, w);
}

auto MetricsCollector::isInAnalysisRegion(const glm::vec3& position,
                                          const BenchmarkResult::SimulationConfig& config,
                                          cuda::Simulation::Parameters::TestCase experimentType) -> bool
{
    switch (experimentType)
    {
    case cuda::Simulation::Parameters::TestCase::PoiseuilleFlow:
    {
        // Middle 10% of channel length (x-direction)
        const auto domainLength = config.domainMax.x - config.domainMin.x;
        const auto middleStart = config.domainMin.x + 0.45F * domainLength;
        const auto middleEnd = config.domainMin.x + 0.55F * domainLength;
        return position.x >= middleStart && position.x <= middleEnd;
    }

    case cuda::Simulation::Parameters::TestCase::TaylorGreenVortex:
        // Use entire domain for Taylor-Green
        return true;

    default:
        return false;
    }
}

auto MetricsCollector::calculateL2VelocityErrorNorm(const BenchmarkResult::SimulationConfig& config,
                                                    cuda::Simulation::Parameters::TestCase experimentType) const
    -> float
{
    if (_velocitySamples.empty())
    {
        return 0.0F;
    }

    auto sumErrorSquared = 0.0F;
    auto validSamples = 0;
    const auto currentTime = static_cast<float>(_frameCounter) * 0.001F;  // Approximate time

    for (const auto& [position, numericalVel] : _velocitySamples)
    {
        const auto analyticalVel = getAnalyticalVelocity(position, currentTime, config, experimentType);
        const auto error = numericalVel - analyticalVel;
        sumErrorSquared += glm::dot(error, error);
        validSamples++;
    }

    if (validSamples > 0)
    {
        return std::sqrt(sumErrorSquared / static_cast<float>(validSamples));
    }

    return 0.0F;
}

// Profile extraction methods implementation
auto MetricsCollector::extractVelocityProfileVertical(const cuda::Simulation& simulation,
                                                      float xPosition,
                                                      float timeStamp,
                                                      const BenchmarkResult::SimulationConfig& config) const
    -> VelocityProfile
{
    VelocityProfile profile;
    profile.xPosition = xPosition;
    profile.timeStamp = timeStamp;
    profile.profileType = "vertical";

    const auto particleData = simulation.getParticleDataSnapshot();
    const auto xTolerance = 0.1F;
    std::vector<std::pair<float, float>> yVelPairs;

    for (uint32_t i = 0; i < particleData.particleCount; ++i)
    {
        const auto& position = particleData.positions[i];
        const auto& velocity = particleData.velocities[i];

        if (std::abs(position.x - xPosition) <= xTolerance)
        {
            yVelPairs.emplace_back(position.y, velocity.x);  // U component
        }
    }

    std::sort(yVelPairs.begin(), yVelPairs.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    profile.yPositions.reserve(yVelPairs.size());
    profile.xVelocities.reserve(yVelPairs.size());

    for (const auto& [y, vx] : yVelPairs)
    {
        profile.yPositions.push_back(y);
        profile.xVelocities.push_back(vx);
    }

    return profile;
}

auto MetricsCollector::extractVelocityProfileHorizontal(const cuda::Simulation& simulation,
                                                        float yPosition,
                                                        float timeStamp,
                                                        const BenchmarkResult::SimulationConfig& config) const
    -> VelocityProfile
{
    VelocityProfile profile;
    profile.xPosition = yPosition;  // Store y-position in xPosition field
    profile.timeStamp = timeStamp;
    profile.profileType = "horizontal";

    const auto particleData = simulation.getParticleDataSnapshot();
    const auto yTolerance = 0.1F;
    std::vector<std::pair<float, float>> xVelPairs;

    for (uint32_t i = 0; i < particleData.particleCount; ++i)
    {
        const auto& position = particleData.positions[i];
        const auto& velocity = particleData.velocities[i];

        if (std::abs(position.y - yPosition) <= yTolerance)
        {
            xVelPairs.emplace_back(position.x, velocity.y);  // V component
        }
    }

    std::sort(xVelPairs.begin(), xVelPairs.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    profile.yPositions.reserve(xVelPairs.size());   // Actually x-positions
    profile.xVelocities.reserve(xVelPairs.size());  // Actually y-velocities

    for (const auto& [x, vy] : xVelPairs)
    {
        profile.yPositions.push_back(x);    // X positions in yPositions field
        profile.xVelocities.push_back(vy);  // V velocity in xVelocities field
    }

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

    const auto particleData = simulation.getParticleDataSnapshot();
    const auto xTolerance = 0.1F;
    std::vector<std::pair<float, float>> yDensPairs;

    for (uint32_t i = 0; i < particleData.particleCount; ++i)
    {
        const auto& position = particleData.positions[i];
        const auto density = particleData.densities[i];

        if (std::abs(position.x - xPosition) <= xTolerance)
        {
            yDensPairs.emplace_back(position.y, density);
        }
    }

    std::sort(yDensPairs.begin(), yDensPairs.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    profile.yPositions.reserve(yDensPairs.size());
    profile.densities.reserve(yDensPairs.size());

    for (const auto& [y, density] : yDensPairs)
    {
        profile.yPositions.push_back(y);
        profile.densities.push_back(density);
    }

    return profile;
}

// Save to file implementation
void MetricsCollector::saveToFile(const BenchmarkResult& result, const std::string& outputPath)
{
    using json = nlohmann::json;

    json resultJson;
    resultJson["experimentType"] = static_cast<uint32_t>(result.experimentType);
    resultJson["simulationType"] = static_cast<uint32_t>(result.simulationType);

    // Basic metrics
    resultJson["metrics"]["l2DensityErrorNorm"] = result.l2DensityErrorNorm;
    resultJson["metrics"]["pressureFieldSmoothness"] = result.pressureFieldSmoothness;
    resultJson["metrics"]["massConservationError"] = result.massConservationError;
    resultJson["metrics"]["averageFrameTime"] = result.averageFrameTime;
    resultJson["metrics"]["particleClusteringIndex"] = result.particleClusteringIndex;

    // Time series data
    resultJson["timeSeries"]["frameTimes"] = result.frameTimes;
    resultJson["timeSeries"]["l2DensityErrors"] = result.l2DensityErrors;

    // Enhanced metrics (if available)
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
            profileJson["profileType"] = profile.profileType;
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

    // Particle snapshots
    if (!result.particleSnapshots.empty())
    {
        json snapshotsJson = json::array();
        for (const auto& snapshot : result.particleSnapshots)
        {
            json snapshotJson;
            snapshotJson["timeStamp"] = snapshot.timeStamp;
            snapshotJson["particleCount"] = snapshot.particleCount;

            json positionsJson = json::array();
            for (const auto& pos : snapshot.positions)
            {
                positionsJson.push_back({pos.x, pos.y, pos.z});
            }
            snapshotJson["positions"] = positionsJson;

            json velocitiesJson = json::array();
            for (const auto& vel : snapshot.velocities)
            {
                velocitiesJson.push_back({vel.x, vel.y, vel.z});
            }
            snapshotJson["velocities"] = velocitiesJson;

            snapshotsJson.push_back(snapshotJson);
        }
        resultJson["particleSnapshots"] = snapshotsJson;
    }

    // Simulation configuration
    json configJson;
    configJson["restDensity"] = result.config.restDensity;
    configJson["viscosityConstant"] = result.config.viscosityConstant;
    configJson["domainMin"] = {result.config.domainMin.x, result.config.domainMin.y, result.config.domainMin.z};
    configJson["domainMax"] = {result.config.domainMax.x, result.config.domainMax.y, result.config.domainMax.z};

    // Experiment-specific parameters
    configJson["channelHeight"] = result.config.channelHeight;
    configJson["channelLength"] = result.config.channelLength;
    configJson["channelWidth"] = result.config.channelWidth;
    configJson["forceMagnitude"] = result.config.forceMagnitude;
    configJson["cavitySize"] = result.config.cavitySize;
    configJson["lidVelocity"] = result.config.lidVelocity;
    configJson["domainSize"] = result.config.domainSize;

    resultJson["config"] = configJson;

    // Save to file
    const auto filename = fmt::format("{}/{}_{}.json",
                                      outputPath,
                                      static_cast<uint32_t>(result.experimentType),
                                      static_cast<uint32_t>(result.simulationType));

    std::ofstream file(filename);
    if (file.is_open())
    {
        file << resultJson.dump(4);
        file.close();
        panda::log::Info("Saved unified benchmark results to: {}", filename);
    }
    else
    {
        panda::log::Error("Failed to save benchmark results to: {}", filename);
    }
}

// Implementation of remaining basic methods
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

}
