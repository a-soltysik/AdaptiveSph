// ExperimentBase.cpp
#include "ExperimentBase.hpp"

#include <panda/Logger.h>

#include "Window.hpp"
#include "gui/SimulationDataGui.hpp"
#include "utils/FrameTimeManager.hpp"

namespace sph::benchmark
{

ExperimentBase::ExperimentBase(cuda::Simulation::Parameters::TestCase name)
    : _name(name)
{
}

BenchmarkResult ExperimentBase::runBenchmark(const BenchmarkParameters& params,
                                             BenchmarkResult::SimulationType simulationType,
                                             panda::gfx::vulkan::Context& api,
                                             bool visualize,
                                             Window* window)
{
    // Create simulation parameters based on configuration
    auto simulationParams = createSimulationParameters(params, simulationType);
    // Initialize particles
    std::vector<glm::vec4> particles;
    initializeParticles(particles, simulationParams);
    // Create refinement parameters
    cuda::refinement::RefinementParameters refinementParams;
    refinementParams.enabled = (simulationType == BenchmarkResult::SimulationType::Adaptive);
    refinementParams.maxParticleCount = 1000000;
    refinementParams.initialCooldown = 10000;
    refinementParams.cooldown = 1000;
    refinementParams.maxMassRatio = 1.1;
    refinementParams.minMassRatio = 0.9;
    refinementParams.criterionType = "velocity";
    refinementParams.velocity.merge.maximalSpeedThreshold = 2.F;
    refinementParams.velocity.split.minimalSpeedThreshold = 4.F;
    refinementParams.splitting.alpha = 0.55;
    refinementParams.splitting.epsilon = 0.65;
    refinementParams.splitting.centerMassRatio = 0.2;
    refinementParams.splitting.vertexMassRatio = 0.067;
    auto simulation = cuda::createSimulation(simulationParams,
                                             particles,
                                             api.initializeParticleSystem(refinementParams.maxParticleCount),
                                             refinementParams);
    // Create metrics collector
    MetricsCollector metricsCollector;
    // Create visualizer if visualization is enabled
    std::unique_ptr<BenchmarkVisualizer> visualizer;
    if (visualize)
    {
        visualizer = std::make_unique<BenchmarkVisualizer>(api, _name, simulationType);
        visualizer->initialize(simulationParams);
    }
    // Run simulation and collect metrics
    runSimulation(*simulation,
                  simulationParams,
                  metricsCollector,
                  params.totalSimulationFrames,
                  params.measurementInterval,
                  visualize ? visualizer.get() : nullptr,
                  window);
    // Calculate and return results
    return metricsCollector.calculateResults(_name, simulationType, params.reynoldsNumber);
}

void ExperimentBase::runSimulation(cuda::Simulation& simulation,
                                   const cuda::Simulation::Parameters& simulationParams,
                                   MetricsCollector& metricsCollector,
                                   int totalFrames,
                                   int measureInterval,
                                   BenchmarkVisualizer* visualizer,
                                   Window* window)
{
    FrameTimeManager timeManager;
    // Initialize metrics collector with simulation data
    metricsCollector.initialize(simulation, simulationParams.restDensity);
    panda::log::Info("Running simulation for {} frames, measuring every {} frames", totalFrames, measureInterval);
    // Run simulation for specified frames
    SimulationDataGui gui {*window, {}};
    for (int frame = 0; frame < totalFrames; ++frame)
    {
        timeManager.update();
        float deltaTime = timeManager.getDelta();

        if (window)
        {
            window->processInput();  // This processes GLFW events
        }
        // Run one simulation step (use fixed time step for consistent physics)
        simulation.update(simulationParams, 0.0001f);
        gui.setAverageNeighbourCount(simulation.calculateAverageNeighborCount());
        gui.setDensityDeviation({.densityDeviations = simulation.updateDensityDeviations(),
                                 .particleCount = simulation.getParticlesCount(),
                                 .restDensity = 1000.F});
        // Collect metrics at specified intervals
        if (frame % measureInterval == 0)
        {
            metricsCollector.collectFrameMetrics(simulation, deltaTime);
        }
        // Update visualization if enabled
        if (visualizer)
        {
            visualizer->renderFrame(simulation, deltaTime);
        }
        // Log progress at a reasonable frequency
        if (frame % 100 == 0 || frame == totalFrames - 1)
        {
            panda::log::Info("Completed frame {}/{} ({}%)",
                             frame,
                             totalFrames,
                             static_cast<int>((frame + 1) * 100.0f / totalFrames));
        }
    }

    panda::log::Info("Simulation completed");
}

}  // namespace sph::benchmark
