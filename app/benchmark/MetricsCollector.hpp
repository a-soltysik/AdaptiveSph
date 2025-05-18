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
    float l2DensityErrorNorm = 0.F;
    float pressureFieldSmoothness = 0.F;
    float massConservationError = 0.F;
    float averageFrameTime = 0.F;
    float particleClusteringIndex = 0.F;
    std::vector<float> frameTimes;
    std::vector<float> l2DensityErrors;
    std::vector<float> totalMasses;
};

class MetricsCollector
{
public:
    void reset();
    void initialize(const cuda::Simulation& simulation, float restDensity);
    void collectFrameMetrics(const cuda::Simulation& simulation, float frameTime);
    [[nodiscard]] auto calculateResults(cuda::Simulation::Parameters::TestCase experimentType,
                                        BenchmarkResult::SimulationType simulationType) const -> BenchmarkResult;
    static void saveToFile(const BenchmarkResult& result, const std::string& outputPath);

private:
    [[nodiscard]] auto calculateL2DensityErrorNorm() const -> float;
    [[nodiscard]] static auto calculatePressureFieldSmoothness() -> float;
    [[nodiscard]] auto calculateMassConservationError() const -> float;
    [[nodiscard]] static auto calculateParticleClusteringIndex() -> float;
    [[nodiscard]] auto calculateAverageFrameTime() const -> float;

    float _initialTotalMass {};
    float _restDensity {};
    std::vector<float> _frameTimes;
    std::vector<float> _l2DensityErrors;
    std::vector<float> _totalMasses;
};

}
