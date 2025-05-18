#include "MetricsCollector.hpp"

#include <fmt/format.h>
#include <panda/Logger.h>

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

void MetricsCollector::saveToFile(const BenchmarkResult& result, const std::string& outputPath)
{
    using json = nlohmann::json;

    // Create JSON object
    json resultJson;
    resultJson["experimentType"] = result.experimentType;
    resultJson["simulationType"] = result.simulationType;
    resultJson["metrics"]["l2DensityErrorNorm"] = result.l2DensityErrorNorm;
    resultJson["metrics"]["pressureFieldSmoothness"] = result.pressureFieldSmoothness;
    resultJson["metrics"]["massConservationError"] = result.massConservationError;
    resultJson["metrics"]["averageFrameTime"] = result.averageFrameTime;
    resultJson["metrics"]["particleClusteringIndex"] = result.particleClusteringIndex;

    resultJson["timeSeries"]["frameTimes"] = result.frameTimes;
    resultJson["timeSeries"]["l2DensityErrors"] = result.l2DensityErrors;
    resultJson["timeSeries"]["totalMasses"] = result.totalMasses;

    const auto filename = fmt::format("{}/{}.json",
                                      outputPath,
                                      std::to_underlying(result.experimentType),
                                      std::to_underlying(result.simulationType));

    std::ofstream file(filename);
    if (file.is_open())
    {
        file << resultJson.dump(4);
        file.close();
        panda::log::Info("Saved benchmark results to: {}", filename);
    }
    else
    {
        panda::log::Error("Failed to save benchmark results to: {}", filename);
    }
}

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

}
