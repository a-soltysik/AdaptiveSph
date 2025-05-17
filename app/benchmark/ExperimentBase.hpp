#pragma once
#include <cuda/Simulation.cuh>
#include <memory>

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
    ExperimentBase(cuda::Simulation::Parameters::TestCase name);
    virtual ~ExperimentBase() = default;

    // Initialize and run a benchmark experiment
    virtual BenchmarkResult runBenchmark(const BenchmarkParameters& params,
                                         const cuda::Simulation::Parameters& simulationParameters,
                                         BenchmarkResult::SimulationType simulationType,
                                         panda::gfx::vulkan::Context& api,
                                         bool visualize = true,
                                         Window* window = nullptr);

    // Get the name of the experiment
    cuda::Simulation::Parameters::TestCase getName() const
    {
        return _name;
    }

protected:
    // Create simulation parameters for the experiment
    virtual cuda::Simulation::Parameters createSimulationParameters(
        const BenchmarkParameters& params,
        const cuda::Simulation::Parameters& simulationParameters,
        BenchmarkResult::SimulationType simulationType) = 0;

    // Initialize particles for the experiment
    virtual void initializeParticles(std::vector<glm::vec4>& particles,
                                     const cuda::Simulation::Parameters& simulationParams) = 0;

    // Run simulation for specified frames and collect metrics
    virtual void runSimulation(cuda::Simulation& simulation,
                               const cuda::Simulation::Parameters& simulationParams,
                               MetricsCollector& metricsCollector,
                               int totalFrames,
                               int measureInterval,
                               float timestep,
                               BenchmarkVisualizer* visualizer = nullptr,
                               Window* window = nullptr);

    static void initializeParticlesGrid(std::vector<glm::vec4>& particles,
                                        const cuda::Simulation::Parameters& simulationParams,
                                        const std::string& experimentName);

    cuda::Simulation::Parameters::TestCase _name;
};

}  // namespace sph::benchmark
