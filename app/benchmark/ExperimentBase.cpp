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
                                             const cuda::Simulation::Parameters& simulationParameters,
                                             BenchmarkResult::SimulationType simulationType,
                                             panda::gfx::vulkan::Context& api,
                                             bool visualize,
                                             Window* window)
{
    // Create simulation parameters based on configuration
    auto simulationParams = createSimulationParameters(params, simulationParameters, simulationType);
    // Initialize particles
    std::vector<glm::vec4> particles;
    initializeParticles(particles, simulationParams);
    // Create refinement parameters based on global config
    cuda::refinement::RefinementParameters refinementParams = params.refinement;
    // Only enable refinement for adaptive simulation type
    refinementParams.enabled = (simulationType == BenchmarkResult::SimulationType::Adaptive);
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
                  params.timestep,
                  visualize ? visualizer.get() : nullptr,
                  window);

    // Calculate and return results
    return metricsCollector.calculateResults(_name, simulationType);
}

void ExperimentBase::runSimulation(cuda::Simulation& simulation,
                                   const cuda::Simulation::Parameters& simulationParams,
                                   MetricsCollector& metricsCollector,
                                   int totalFrames,
                                   int measureInterval,
                                   float timestep,
                                   BenchmarkVisualizer* visualizer,
                                   Window* window)
{
    FrameTimeManager timeManager;
    // Initialize metrics collector with simulation data
    metricsCollector.initialize(simulation, simulationParams.restDensity);
    panda::log::Info("Running simulation for {} frames, measuring every {} frames", totalFrames, measureInterval);
    // Run simulation for specified frames
    SimulationDataGui gui {*window};
    for (int frame = 0; frame < totalFrames; ++frame)
    {
        timeManager.update();
        float deltaTime = timeManager.getDelta();

        if (window)
        {
            window->processInput();  // This processes GLFW events
        }
        // Run one simulation step with configurable time step
        simulation.update(timestep);
        gui.setAverageNeighbourCount(simulation.calculateAverageNeighborCount());
        gui.setDensityDeviation({.densityDeviations = simulation.updateDensityDeviations(),
                                 .particleCount = simulation.getParticlesCount(),
                                 .restDensity = simulationParams.restDensity});
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

void ExperimentBase::initializeParticlesGrid(std::vector<glm::vec4>& particles,
                                             const cuda::Simulation::Parameters& simulationParams,
                                             const std::string& experimentName)
{
    // Clear any existing particles
    particles.clear();

    // Calculate the domain size
    const glm::vec3 domainMin = simulationParams.domain.min;
    const glm::vec3 domainMax = simulationParams.domain.max;
    const glm::vec3 domainSize = domainMax - domainMin;

    // Calculate particle spacing based on particle radius
    float particleSpacing = simulationParams.baseParticleRadius * 2.0f;

    // Calculate number of particles in each dimension
    glm::uvec3 gridSize;
    gridSize.x = static_cast<uint32_t>(std::floor(domainSize.x / particleSpacing));
    gridSize.y = static_cast<uint32_t>(std::floor(domainSize.y / particleSpacing));
    gridSize.z = static_cast<uint32_t>(std::floor(domainSize.z / particleSpacing));

    panda::log::Info("Creating {} with grid size: {}x{}x{}", experimentName, gridSize.x, gridSize.y, gridSize.z);

    // Calculate actual spacing to distribute particles evenly
    glm::vec3 actualSpacing;
    actualSpacing.x = domainSize.x / static_cast<float>(gridSize.x);
    actualSpacing.y = domainSize.y / static_cast<float>(gridSize.y);
    actualSpacing.z = domainSize.z / static_cast<float>(gridSize.z);

    // Calculate offset to center particles within the domain
    glm::vec3 offset = actualSpacing * 0.5f;

    // Create particles throughout the domain
    for (uint32_t i = 0; i < gridSize.x; i++)
    {
        for (uint32_t j = 0; j < gridSize.y; j++)
        {
            for (uint32_t k = 0; k < gridSize.z; k++)
            {
                // Calculate position with offset to avoid domain boundaries
                float x = domainMin.x + i * actualSpacing.x + offset.x;
                float y = domainMin.y + j * actualSpacing.y + offset.y;
                float z = domainMin.z + k * actualSpacing.z + offset.z;

                const auto position = glm::vec3(x, y, z);
                particles.emplace_back(position, 0.0f);
            }
        }
    }

    panda::log::Info("Created {} particles for {}", particles.size(), experimentName);
    panda::log::Info("Particle spacing: {}, {}, {}", actualSpacing.x, actualSpacing.y, actualSpacing.z);
    panda::log::Info("First particle at: {}, {}, {}",
                     domainMin.x + offset.x,
                     domainMin.y + offset.y,
                     domainMin.z + offset.z);
}

}  // namespace sph::benchmark
