#pragma once
#include <cstdint>
#include <cuda/Simulation.cuh>
#include <glm/ext/vector_float4.hpp>
#include <string>
#include <vector>

#include "BenchmarkVisualizer.hpp"
#include "MetricsCollector.hpp"
#include "Window.hpp"
#include "panda/gfx/vulkan/Context.h"
#include "utils/ConfigurationManager.hpp"

namespace sph::benchmark
{

class ExperimentBase
{
public:
    explicit ExperimentBase(cuda::Simulation::Parameters::TestCase name);
    virtual ~ExperimentBase() = default;
    auto runBenchmark(const BenchmarkParameters& params,
                      const cuda::Simulation::Parameters& simulationParameters,
                      BenchmarkResult::SimulationType simulationType,
                      panda::gfx::vulkan::Context& api,
                      bool visualize = true,
                      Window* window = nullptr) -> BenchmarkResult;

    [[nodiscard]] auto getName() const -> cuda::Simulation::Parameters::TestCase
    {
        return _name;
    }

protected:
    virtual auto createSimulationParameters(const BenchmarkParameters& params,
                                            const cuda::Simulation::Parameters& simulationParameters,
                                            BenchmarkResult::SimulationType simulationType)
        -> cuda::Simulation::Parameters = 0;

    virtual auto initializeParticles(const cuda::Simulation::Parameters& simulationParams)
        -> std::vector<glm::vec4> = 0;

    virtual void runSimulation(cuda::Simulation& simulation,
                               const cuda::Simulation::Parameters& simulationParams,
                               MetricsCollector& metricsCollector,
                               uint32_t totalFrames,
                               uint32_t measureInterval,
                               float timestep,
                               BenchmarkVisualizer* visualizer = nullptr,
                               Window* window = nullptr);

    static auto initializeParticlesGrid(const cuda::Simulation::Parameters& simulationParams,
                                        const std::string& experimentName) -> std::vector<glm::vec4>;

private:
    cuda::Simulation::Parameters::TestCase _name;
};

}
