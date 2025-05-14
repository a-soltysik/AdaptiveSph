// MetricsCollector.hpp
#pragma once
#include <cuda/Simulation.cuh>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace sph::benchmark
{

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
    float reynoldsNumber;
    // Metrics
    float l2DensityErrorNorm;
    float pressureFieldSmoothness;
    float massConservationError;
    float averageFrameTime;
    float particleClusteringIndex;
    // Collected data for time series analysis
    std::vector<float> frameTimes;
    std::vector<float> l2DensityErrors;
    std::vector<float> totalMasses;
};

class MetricsCollector
{
public:
    MetricsCollector();
    // Reset for a new experiment
    void reset();
    // Initialize with starting state of simulation
    void initialize(const cuda::Simulation& simulation, float restDensity);
    // Collect frame metrics
    void collectFrameMetrics(const cuda::Simulation& simulation, float frameTime);
    // Calculate and return final results
    BenchmarkResult calculateResults(cuda::Simulation::Parameters::TestCase experimentType,
                                     BenchmarkResult::SimulationType simulationType,
                                     float reynoldsNumber) const;
    // Save results to file
    void saveToFile(const BenchmarkResult& result, const std::string& outputPath) const;

private:
    float _initialTotalMass;
    float _restDensity;
    std::vector<float> _frameTimes;
    std::vector<float> _l2DensityErrors;
    std::vector<float> _totalMasses;

    // Calculate specific metrics
    float calculateL2DensityErrorNorm() const;
    float calculatePressureFieldSmoothness() const;
    float calculateMassConservationError() const;
    float calculateParticleClusteringIndex() const;
    float calculateAverageFrameTime() const;
};

}  // namespace sph::benchmark
