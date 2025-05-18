#pragma once
#include <glm/ext/vector_float4.hpp>
#include <vector>

#include "ExperimentBase.hpp"
#include "Window.hpp"
#include "benchmark/MetricsCollector.hpp"
#include "cuda/Simulation.cuh"
#include "panda/gfx/vulkan/Context.h"
#include "utils/ConfigurationManager.hpp"

namespace sph::benchmark
{

class PoiseuilleFlow : public ExperimentBase
{
public:
    PoiseuilleFlow();

protected:
    auto createSimulationParameters(const BenchmarkParameters& params,
                                    const cuda::Simulation::Parameters& simulationParameters,
                                    BenchmarkResult::SimulationType simulationType)
        -> cuda::Simulation::Parameters override;

    auto initializeParticles(const cuda::Simulation::Parameters& simulationParams) -> std::vector<glm::vec4> override;
};

}
