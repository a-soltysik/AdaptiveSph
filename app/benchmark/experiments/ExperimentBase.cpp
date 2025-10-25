#include "ExperimentBase.hpp"

#include <panda/Logger.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/ext/vector_uint3.hpp>
#include <memory>
#include <string>
#include <vector>

#include "benchmark/BenchmarkVisualizer.hpp"
#include "benchmark/MetricsCollector.hpp"
#include "cuda/Simulation.cuh"
#include "panda/gfx/vulkan/Context.h"
#include "ui/SimulationDataGui.hpp"
#include "ui/Window.hpp"
#include "utils/ConfigurationManager.hpp"
#include "utils/FrameTimeManager.hpp"

namespace sph::benchmark
{

ExperimentBase::ExperimentBase(cuda::Simulation::Parameters::TestCase name)
    : _name(name)
{
}

auto ExperimentBase::runBenchmark(const BenchmarkParameters& params,
                                  const cuda::Simulation::Parameters& simulationParameters,
                                  BenchmarkResult::SimulationType simulationType,
                                  panda::gfx::vulkan::Context& api,
                                  Window& window,
                                  bool visualize) -> BenchmarkResult
{
    // Create simulation parameters and configuration
    auto simulationParams = createSimulationParameters(params, simulationParameters, simulationType);
    const auto particles = initializeParticles(simulationParams);
    const auto config = createBenchmarkConfig(params, simulationParams);
    // Configure refinement for adaptive simulations
    auto refinementParams = params.refinement;
    refinementParams.enabled = (simulationType == BenchmarkResult::SimulationType::Adaptive);
    // Create simulation
    auto simulation = cuda::createSimulation(simulationParams,
                                             particles,
                                             api.initializeParticleSystem(refinementParams.maxParticleCount),
                                             refinementParams);

    // Initialize metrics collector
    MetricsCollector metricsCollector;
    // Run simulation with or without visualization
    if (visualize)
    {
        auto visualizer = std::make_unique<BenchmarkVisualizer>(api, _name, simulationType);
        visualizer->initialize(simulationParams);
        runSimulation(*simulation,
                      simulationParams,
                      metricsCollector,
                      params.totalSimulationFrames,
                      params.measurementInterval,
                      params.timestep,
                      window,
                      visualizer.get());
    }
    else
    {
        runSimulation(*simulation,
                      simulationParams,
                      metricsCollector,
                      params.totalSimulationFrames,
                      params.measurementInterval,
                      params.timestep,
                      window,
                      nullptr);
    }

    // Calculate and return results with enhanced metrics
    return metricsCollector.calculateResults(_name, simulationType, config);
}

void ExperimentBase::runSimulation(cuda::Simulation& simulation,
                                   const cuda::Simulation::Parameters& simulationParams,
                                   MetricsCollector& metricsCollector,
                                   uint32_t totalFrames,
                                   uint32_t measureInterval,
                                   float timestep,
                                   Window& window,
                                   BenchmarkVisualizer* visualizer)
{
    // Create unified benchmark configuration
    const auto config = createBenchmarkConfig(BenchmarkParameters {}, simulationParams);
    FrameTimeManager timeManager;
    metricsCollector.initialize(simulation, simulationParams.restDensity);
    panda::log::Info("Running {} simulation for {} frames, measuring every {} frames",
                     static_cast<uint32_t>(_name),
                     totalFrames,
                     measureInterval);

    SimulationDataGui gui {};
    for (uint32_t frame = 0; frame < totalFrames; ++frame)
    {
        timeManager.update();
        const auto deltaTime = timeManager.getDelta();
        // Handle window events
        window.processInput();
        if (window.shouldClose())
        {
            panda::log::Warning("Simulation has been stopped");
            return;
        }

        // Update simulation
        simulation.update(timestep);
        // Update GUI
        gui.setAverageNeighbourCount(simulation.calculateAverageNeighborCount());
        gui.setDensityDeviation({.densityDeviations = simulation.updateDensityDeviations(),
                                 .particleCount = simulation.getParticlesCount(),
                                 .restDensity = simulationParams.restDensity});

        // Collect metrics at measurement intervals
        if (frame % measureInterval == 0)
        {
            const auto cudaTime = simulation.getLastCudaComputationTime();
            // Use unified enhanced metrics collection for all experiments
            metricsCollector.collectEnhancedMetrics(simulation, deltaTime, cudaTime, config, _name);
        }

        // Render frame if visualizer is available
        if (visualizer != nullptr)
        {
            visualizer->renderFrame(simulation);
        }

        // Progress logging
        if (frame % 100 == 0 || frame == totalFrames - 1)
        {
            panda::log::Info("Completed frame {}/{} ({}%)", frame, totalFrames, (frame + 1) * 100 / totalFrames);
        }
    }

    panda::log::Info("{} simulation completed", static_cast<uint32_t>(_name));
}

auto ExperimentBase::initializeParticlesGrid(const cuda::Simulation::Parameters& simulationParams,
                                             const std::string& experimentName) -> std::vector<glm::vec4>
{
    const auto domainMin = simulationParams.domain.min;
    const auto domainMax = simulationParams.domain.max;
    const auto domainSize = domainMax - domainMin;

    const auto particleSpacing = simulationParams.baseParticleRadius * 2.0F;
    const auto gridSize = glm::uvec3 {static_cast<uint32_t>(std::floor(domainSize.x / particleSpacing)),
                                      static_cast<uint32_t>(std::floor(domainSize.y / particleSpacing)),
                                      static_cast<uint32_t>(std::floor(domainSize.z / particleSpacing))};

    panda::log::Info("Creating {} with grid size: {}x{}x{}", experimentName, gridSize.x, gridSize.y, gridSize.z);

    const auto actualSpacing = glm::vec3 {domainSize.x / static_cast<float>(gridSize.x),
                                          domainSize.y / static_cast<float>(gridSize.y),
                                          domainSize.z / static_cast<float>(gridSize.z)};

    const auto offset = actualSpacing * 0.5F;

    auto result = std::vector<glm::vec4>();
    result.reserve(static_cast<size_t>(gridSize.x) * gridSize.y * gridSize.z);

    for (uint32_t i = 0; i < gridSize.x; i++)
    {
        for (uint32_t j = 0; j < gridSize.y; j++)
        {
            for (uint32_t k = 0; k < gridSize.z; k++)
            {
                const auto x = domainMin.x + (static_cast<float>(i) * actualSpacing.x) + offset.x;
                const auto y = domainMin.y + (static_cast<float>(j) * actualSpacing.y) + offset.y;
                const auto z = domainMin.z + (static_cast<float>(k) * actualSpacing.z) + offset.z;

                const auto position = glm::vec3(x, y, z);
                result.emplace_back(position, 0.0F);
            }
        }
    }

    panda::log::Info("Created {} particles for {}", result.size(), experimentName);
    panda::log::Info("Particle spacing: {}, {}, {}", actualSpacing.x, actualSpacing.y, actualSpacing.z);
    panda::log::Info("First particle at: {}, {}, {}",
                     domainMin.x + offset.x,
                     domainMin.y + offset.y,
                     domainMin.z + offset.z);

    return result;
}

}
