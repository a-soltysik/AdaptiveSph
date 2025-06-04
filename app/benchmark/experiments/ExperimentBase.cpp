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
    auto simulationParams = createSimulationParameters(params, simulationParameters, simulationType);
    const auto particles = initializeParticles(simulationParams);
    auto refinementParams = params.refinement;
    refinementParams.enabled = (simulationType == BenchmarkResult::SimulationType::Adaptive);
    auto simulation = cuda::createSimulation(simulationParams,
                                             particles,
                                             api.initializeParticleSystem(refinementParams.maxParticleCount),
                                             refinementParams);
    MetricsCollector metricsCollector;
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

    // NEW: Use enhanced metrics calculation if supported
    if (supportsEnhancedMetrics())
    {
        const auto config = createBenchmarkConfig(params, simulationParams);
        return metricsCollector.calculateResults(_name, simulationType, config);
    }
    else
    {
        return metricsCollector.calculateResults(_name, simulationType);
    }
}

//NOLINTBEGIN(bugprone-easily-swappable-parameters)
void ExperimentBase::runSimulation(cuda::Simulation& simulation,
                                   const cuda::Simulation::Parameters& simulationParams,
                                   MetricsCollector& metricsCollector,
                                   uint32_t totalFrames,
                                   uint32_t measureInterval,
                                   float timestep,
                                   Window& window,
                                   BenchmarkVisualizer* visualizer)
//NOLINTEND(bugprone-easily-swappable-parameters)
{
    BenchmarkResult::SimulationConfig config;
    config.cavitySize = simulationParams.domain.max.x - simulationParams.domain.min.x;
    config.lidVelocity = simulationParams.lidVelocity;
    config.restDensity = simulationParams.restDensity;
    config.viscosityConstant = simulationParams.viscosityConstant;
    config.domainMin = simulationParams.domain.min;
    config.domainMax = simulationParams.domain.max;

    FrameTimeManager timeManager;
    metricsCollector.initialize(simulation, simulationParams.restDensity);
    panda::log::Info("Running simulation for {} frames, measuring every {} frames", totalFrames, measureInterval);
    SimulationDataGui gui {};
    for (uint32_t frame = 0; frame < totalFrames; ++frame)
    {
        timeManager.update();
        const auto deltaTime = timeManager.getDelta();
        window.processInput();
        if (window.shouldClose())
        {
            panda::log::Warning("Simulation has been stopped");
            return;
        }

        simulation.update(timestep);
        gui.setAverageNeighbourCount(simulation.calculateAverageNeighborCount());
        gui.setDensityDeviation({.densityDeviations = simulation.updateDensityDeviations(),
                                 .particleCount = simulation.getParticlesCount(),
                                 .restDensity = simulationParams.restDensity});
        if (frame % measureInterval == 0)
        {
            const auto cudaTime = simulation.getLastCudaComputationTime();
            // KLUCZOWE: wywołanie collectCavityMetrics zamiast collectFrameMetrics
            metricsCollector.collectCavityMetrics(simulation, deltaTime, cudaTime, config);
        }
        if (visualizer != nullptr)
        {
            visualizer->renderFrame(simulation);
        }
        if (frame % 100 == 0 || frame == totalFrames - 1)
        {
            panda::log::Info("Completed frame {}/{} ({}%)", frame, totalFrames, (frame + 1) * 100 / totalFrames);
        }
    }

    panda::log::Info("Simulation completed");
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
