#pragma once
#include <glm/ext/vector_float4.hpp>
#include <vector>

#include "../../ui/Window.hpp"
#include "ExperimentBase.hpp"
#include "benchmark/MetricsCollector.hpp"
#include "cuda/Simulation.cuh"
#include "panda/gfx/vulkan/Context.h"
#include "ui/SimulationDataGui.hpp"
#include "utils/ConfigurationManager.hpp"
#include "utils/FrameTimeManager.hpp"

namespace sph::benchmark
{

class LidDrivenCavity : public ExperimentBase
{
public:
    LidDrivenCavity();

protected:
    cuda::Simulation::Parameters createSimulationParameters(const BenchmarkParameters& params,
                                                            const cuda::Simulation::Parameters& simulationParameters,
                                                            BenchmarkResult::SimulationType simulationType) override;

    auto initializeParticles(const cuda::Simulation::Parameters& simulationParams) -> std::vector<glm::vec4> override;

    // NEW: Create configuration for enhanced metrics
    [[nodiscard]] auto createBenchmarkConfig(const BenchmarkParameters& params,
                                             const cuda::Simulation::Parameters& simulationParams) const
        -> BenchmarkResult::SimulationConfig override;

    // NEW: Enable enhanced metrics for Lid Driven Cavity
    [[nodiscard]] auto supportsEnhancedMetrics() const -> bool override;

    // NEW: Override runSimulation for Cavity-specific enhanced metrics collection
    void runSimulation(cuda::Simulation& simulation,
                       const cuda::Simulation::Parameters& simulationParams,
                       MetricsCollector& metricsCollector,
                       uint32_t totalFrames,
                       uint32_t measureInterval,
                       float timestep,
                       Window& window,
                       BenchmarkVisualizer* visualizer = nullptr) override;
};

}
