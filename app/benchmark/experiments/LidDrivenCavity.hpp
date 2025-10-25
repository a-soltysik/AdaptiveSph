#pragma once
#include <glm/ext/vector_float4.hpp>
#include <vector>

#include "ExperimentBase.hpp"
#include "cuda/Simulation.cuh"
#include "utils/ConfigurationManager.hpp"

namespace sph::benchmark
{

class LidDrivenCavity : public ExperimentBase
{
public:
    LidDrivenCavity();

protected:
    auto createSimulationParameters(const BenchmarkParameters& params,
                                    const cuda::Simulation::Parameters& simulationParameters,
                                    BenchmarkResult::SimulationType simulationType)
        -> cuda::Simulation::Parameters override;

    auto initializeParticles(const cuda::Simulation::Parameters& simulationParams) -> std::vector<glm::vec4> override;

    auto createBenchmarkConfig(const BenchmarkParameters& params,
                               const cuda::Simulation::Parameters& simulationParams) const
        -> BenchmarkResult::SimulationConfig override;
};

}
