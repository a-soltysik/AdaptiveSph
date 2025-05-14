// MetricsCollector.cpp
#include "MetricsCollector.hpp"

#include <panda/Logger.h>

#include <cmath>
#include <fstream>
#include <numeric>

namespace sph::benchmark
{

MetricsCollector::MetricsCollector()
    : _initialTotalMass(0.0f),
      _restDensity(0.0f)
{
}

void MetricsCollector::reset()
{
    _initialTotalMass = 0.0f;
    _restDensity = 0.0f;
    _frameTimes.clear();
    _l2DensityErrors.clear();
    _totalMasses.clear();
}

void MetricsCollector::initialize(const cuda::Simulation& simulation, float restDensity)
{
    _restDensity = restDensity;

    // Calculate initial total mass (for mass conservation tracking)
    // For a proper implementation, we would need access to particle masses
    auto particleCount = simulation.getParticlesCount();
    if (particleCount > 0)
    {
        // This is a placeholder - ideally we would sum all particle masses
        _initialTotalMass = particleCount * restDensity;
    }
}

void MetricsCollector::collectFrameMetrics(const cuda::Simulation& simulation, float frameTime)
{
    // Store frame time
    _frameTimes.push_back(frameTime);

    // Calculate L2 density error norm
    auto densityDeviations = simulation.updateDensityDeviations();
    auto particleCount = simulation.getParticlesCount();

    if (particleCount > 0)
    {
        // Calculate L2 norm of density errors
        float sumErrorSquared = 0.0f;
        for (uint32_t i = 0; i < particleCount; ++i)
        {
            float error = densityDeviations[i].x;  // x component has normalized density deviation
            sumErrorSquared += error * error;
        }
        float l2Norm = std::sqrt(sumErrorSquared / particleCount);
        _l2DensityErrors.push_back(l2Norm);

        // Calculate total mass (simplified - would need actual masses)
        float currentTotalMass = particleCount * _restDensity;
        _totalMasses.push_back(currentTotalMass);
    }
}

BenchmarkResult MetricsCollector::calculateResults(cuda::Simulation::Parameters::TestCase experimentType,
                                                   BenchmarkResult::SimulationType simulationType,
                                                   float reynoldsNumber) const
{
    BenchmarkResult result;
    result.experimentType = experimentType;
    result.simulationType = simulationType;
    result.reynoldsNumber = reynoldsNumber;

    // Calculate metrics
    result.l2DensityErrorNorm = calculateL2DensityErrorNorm();
    result.pressureFieldSmoothness = calculatePressureFieldSmoothness();
    result.massConservationError = calculateMassConservationError();
    result.averageFrameTime = calculateAverageFrameTime();
    result.particleClusteringIndex = calculateParticleClusteringIndex();

    // Store time series data
    result.frameTimes = _frameTimes;
    result.l2DensityErrors = _l2DensityErrors;
    result.totalMasses = _totalMasses;

    return result;
}

void MetricsCollector::saveToFile(const BenchmarkResult& result, const std::string& outputPath) const
{
    using json = nlohmann::json;

    // Create JSON object
    json resultJson;
    resultJson["experimentType"] = result.experimentType;
    resultJson["simulationType"] = result.simulationType;
    resultJson["reynoldsNumber"] = result.reynoldsNumber;
    resultJson["metrics"]["l2DensityErrorNorm"] = result.l2DensityErrorNorm;
    resultJson["metrics"]["pressureFieldSmoothness"] = result.pressureFieldSmoothness;
    resultJson["metrics"]["massConservationError"] = result.massConservationError;
    resultJson["metrics"]["averageFrameTime"] = result.averageFrameTime;
    resultJson["metrics"]["particleClusteringIndex"] = result.particleClusteringIndex;

    // Store time series data
    resultJson["timeSeries"]["frameTimes"] = result.frameTimes;
    resultJson["timeSeries"]["l2DensityErrors"] = result.l2DensityErrors;
    resultJson["timeSeries"]["totalMasses"] = result.totalMasses;

    // Create filename
    const auto filename = fmt::format("{}/{}_{}_Re{}.json",
                                      outputPath,
                                      std::to_underlying(result.experimentType),
                                      std::to_underlying(result.simulationType),
                                      result.reynoldsNumber);

    // Write to file
    std::ofstream file(filename);
    if (file.is_open())
    {
        file << resultJson.dump(4);  // Pretty-print with 4-space indentation
        file.close();
        panda::log::Info("Saved benchmark results to: {}", filename);
    }
    else
    {
        panda::log::Error("Failed to save benchmark results to: {}", filename);
    }
}

float MetricsCollector::calculateL2DensityErrorNorm() const
{
    if (_l2DensityErrors.empty())
    {
        return 0.0f;
    }

    // Return average L2 error over all frames (excluding initial frames)
    size_t startIndex = std::min(size_t(10), _l2DensityErrors.size() - 1);
    if (startIndex >= _l2DensityErrors.size())
    {
        return 0.0f;
    }

    return std::accumulate(_l2DensityErrors.begin() + startIndex, _l2DensityErrors.end(), 0.0f) /
           (_l2DensityErrors.size() - startIndex);
}

float MetricsCollector::calculatePressureFieldSmoothness() const
{
    // This would require more detailed pressure field data
    // Placeholder implementation
    return 0.0f;
}

float MetricsCollector::calculateMassConservationError() const
{
    if (_totalMasses.empty() || _initialTotalMass <= 0.0f)
    {
        return 0.0f;
    }

    // Calculate the relative error in total mass
    float finalTotalMass = _totalMasses.back();
    return std::abs(finalTotalMass - _initialTotalMass) / _initialTotalMass;
}

float MetricsCollector::calculateParticleClusteringIndex() const
{
    // This would require spatial distribution data of particles
    // Placeholder implementation
    return 0.0f;
}

float MetricsCollector::calculateAverageFrameTime() const
{
    if (_frameTimes.empty())
    {
        return 0.0f;
    }

    return std::accumulate(_frameTimes.begin(), _frameTimes.end(), 0.0f) / _frameTimes.size();
}

}  // namespace sph::benchmark
