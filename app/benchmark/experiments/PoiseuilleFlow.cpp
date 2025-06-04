#include "PoiseuilleFlow.hpp"

#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <vector>

#include "ExperimentBase.hpp"
#include "benchmark/MetricsCollector.hpp"
#include "cuda/Simulation.cuh"
#include "utils/ConfigurationManager.hpp"

namespace sph::benchmark
{

PoiseuilleFlow::PoiseuilleFlow()
    : ExperimentBase(cuda::Simulation::Parameters::TestCase::PoiseuilleFlow)
{
}

auto PoiseuilleFlow::createSimulationParameters(const BenchmarkParameters& params,
                                                const cuda::Simulation::Parameters& simulationParameters,
                                                BenchmarkResult::SimulationType simulationType)
    -> cuda::Simulation::Parameters
{
    cuda::Simulation::Parameters simulationParams = simulationParameters;
    const auto halfHeight = params.channelHeight / 2.0F;
    const auto halfWidth = params.channelWidth / 2.0F;
    const auto halfLength = params.channelLength / 2.0F;
    simulationParams.domain.min = glm::vec3(-halfLength, -halfHeight, -halfWidth);
    simulationParams.domain.max = glm::vec3(halfLength, halfHeight, halfWidth);
    simulationParams.gravity = glm::vec3(params.forceMagnitude, 0.0F, 0.0F);
    if (simulationType == BenchmarkResult::SimulationType::Coarse)
    {
        simulationParams.baseParticleRadius = params.coarse.baseParticleRadius;
        simulationParams.baseParticleMass = params.coarse.baseParticleMass;
        simulationParams.baseSmoothingRadius = params.coarse.baseSmoothingRadius;
        simulationParams.pressureConstant = params.coarse.pressureConstant;
        simulationParams.nearPressureConstant = params.coarse.nearPressureConstant;
        simulationParams.viscosityConstant = params.coarse.viscosityConstant;
    }
    else if (simulationType == BenchmarkResult::SimulationType::Fine)
    {
        simulationParams.baseParticleRadius = params.fine.baseParticleRadius;
        simulationParams.baseParticleMass = params.fine.baseParticleMass;
        simulationParams.baseSmoothingRadius = params.fine.baseSmoothingRadius;
        simulationParams.pressureConstant = params.fine.pressureConstant;
        simulationParams.nearPressureConstant = params.fine.nearPressureConstant;
        simulationParams.viscosityConstant = params.fine.viscosityConstant;
    }
    else  // adaptive
    {
        simulationParams.baseParticleRadius = params.adaptive.baseParticleRadius;
        simulationParams.baseParticleMass = params.adaptive.baseParticleMass;
        simulationParams.baseSmoothingRadius = params.adaptive.baseSmoothingRadius;
        simulationParams.pressureConstant = params.adaptive.pressureConstant;
        simulationParams.nearPressureConstant = params.adaptive.nearPressureConstant;
        simulationParams.viscosityConstant = params.adaptive.viscosityConstant;
    }

    simulationParams.testCase = cuda::Simulation::Parameters::TestCase::PoiseuilleFlow;

    return simulationParams;
}

auto PoiseuilleFlow::initializeParticles(const cuda::Simulation::Parameters& simulationParams) -> std::vector<glm::vec4>
{
    return initializeParticlesGrid(simulationParams, "Poiseuille Flow");
}

auto PoiseuilleFlow::createBenchmarkConfig(const BenchmarkParameters& params,
                                           const cuda::Simulation::Parameters& simulationParams) const
    -> BenchmarkResult::SimulationConfig
{
    BenchmarkResult::SimulationConfig config;
    // Extract parameters from BenchmarkParameters and SimulationParameters
    config.channelHeight = params.channelHeight;
    config.channelLength = params.channelLength;
    config.channelWidth = params.channelWidth;
    config.forceMagnitude = params.forceMagnitude;
    config.restDensity = simulationParams.restDensity;
    config.viscosityConstant = simulationParams.viscosityConstant;
    config.domainMin = simulationParams.domain.min;
    config.domainMax = simulationParams.domain.max;
    return config;
}

auto PoiseuilleFlow::supportsEnhancedMetrics() const -> bool
{
    return true;
}

void PoiseuilleFlow::runSimulation(cuda::Simulation& simulation,
                                   const cuda::Simulation::Parameters& simulationParams,
                                   MetricsCollector& metricsCollector,
                                   uint32_t totalFrames,
                                   uint32_t measureInterval,
                                   float timestep,
                                   Window& window,
                                   BenchmarkVisualizer* visualizer)
{
    // Create configuration for enhanced Poiseuille metrics
    BenchmarkResult::SimulationConfig config;
    config.restDensity = simulationParams.restDensity;
    config.viscosityConstant = simulationParams.viscosityConstant;
    config.domainMin = simulationParams.domain.min;
    config.domainMax = simulationParams.domain.max;
    // Extract Poiseuille-specific parameters (we need access to BenchmarkParameters here)
    // For now, extract from domain and simulation parameters
    config.channelHeight = simulationParams.domain.max.y - simulationParams.domain.min.y;
    config.channelLength = simulationParams.domain.max.x - simulationParams.domain.min.x;
    config.channelWidth = simulationParams.domain.max.z - simulationParams.domain.min.z;
    config.forceMagnitude = simulationParams.gravity.x;  // Force is applied in x-direction
    FrameTimeManager timeManager;
    metricsCollector.initialize(simulation, simulationParams.restDensity);
    panda::log::Info("Running Poiseuille flow simulation for {} frames, measuring every {} frames",
                     totalFrames,
                     measureInterval);
    SimulationDataGui gui {};
    for (uint32_t frame = 0; frame < totalFrames; ++frame)
    {
        timeManager.update();
        const auto deltaTime = timeManager.getDelta();

        window.processInput();

        simulation.update(timestep);
        gui.setAverageNeighbourCount(simulation.calculateAverageNeighborCount());
        gui.setDensityDeviation({.densityDeviations = simulation.updateDensityDeviations(),
                                 .particleCount = simulation.getParticlesCount(),
                                 .restDensity = simulationParams.restDensity});
        if (frame % measureInterval == 0)
        {
            // Get CUDA computation time from simulation
            const auto cudaTime = simulation.getLastCudaComputationTime();
            // Use enhanced Poiseuille metrics collection with performance timing
            metricsCollector.collectPoiseuilleMetrics(simulation, deltaTime, cudaTime, config);
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

    panda::log::Info("Poiseuille flow simulation completed");
}

}
