#pragma once
#include <cstdint>
#include <cuda/Simulation.cuh>
#include <glm/ext/vector_float4.hpp>
#include <string>
#include <vector>

#include "../../ui/Window.hpp"
#include "../BenchmarkVisualizer.hpp"
#include "../MetricsCollector.hpp"
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
                      Window& window,
                      bool visualize = true) -> BenchmarkResult;

    [[nodiscard]] auto getName() const -> cuda::Simulation::Parameters::TestCase
    {
        return _name;
    }

protected:
    // Pure virtual methods that must be implemented by derived classes
    virtual auto createSimulationParameters(const BenchmarkParameters& params,
                                            const cuda::Simulation::Parameters& simulationParameters,
                                            BenchmarkResult::SimulationType simulationType)
        -> cuda::Simulation::Parameters = 0;

    virtual auto initializeParticles(const cuda::Simulation::Parameters& simulationParams)
        -> std::vector<glm::vec4> = 0;

    virtual auto createBenchmarkConfig(const BenchmarkParameters& params,
                                       const cuda::Simulation::Parameters& simulationParams) const
        -> BenchmarkResult::SimulationConfig = 0;

    // Unified simulation runner for all experiments
    virtual void runSimulation(cuda::Simulation& simulation,
                               const cuda::Simulation::Parameters& simulationParams,
                               MetricsCollector& metricsCollector,
                               uint32_t totalFrames,
                               uint32_t measureInterval,
                               float timestep,
                               Window& window,
                               BenchmarkVisualizer* visualizer = nullptr);

    // All experiments now support enhanced metrics
    [[nodiscard]] virtual auto supportsEnhancedMetrics() const -> bool
    {
        return true;
    }

    // Utility method for grid-based particle initialization
    static auto initializeParticlesGrid(const cuda::Simulation::Parameters& simulationParams,
                                        const std::string& experimentName) -> std::vector<glm::vec4>;

private:
    cuda::Simulation::Parameters::TestCase _name;
};
}
