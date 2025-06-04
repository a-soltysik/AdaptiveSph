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
                               Window& window,
                               BenchmarkVisualizer* visualizer = nullptr);

    // NEW: Virtual method for creating benchmark configuration
    // Default implementation returns empty config, derived classes can override
    [[nodiscard]] virtual auto createBenchmarkConfig(const BenchmarkParameters& params,
                                                     const cuda::Simulation::Parameters& simulationParams) const
        -> BenchmarkResult::SimulationConfig
    {
        // Default empty configuration
        BenchmarkResult::SimulationConfig config;
        config.restDensity = simulationParams.restDensity;
        config.viscosityConstant = simulationParams.viscosityConstant;
        config.domainMin = simulationParams.domain.min;
        config.domainMax = simulationParams.domain.max;
        return config;
    }

    // NEW: Check if this experiment supports enhanced metrics
    [[nodiscard]] virtual auto supportsEnhancedMetrics() const -> bool
    {
        return false;  // Default: no enhanced metrics
    }

    static auto initializeParticlesGrid(const cuda::Simulation::Parameters& simulationParams,
                                        const std::string& experimentName) -> std::vector<glm::vec4>;

private:
    cuda::Simulation::Parameters::TestCase _name;
};

}
