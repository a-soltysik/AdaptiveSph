#include "LidDrivenCavity.hpp"

#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <vector>

#include "ExperimentBase.hpp"
#include "benchmark/MetricsCollector.hpp"
#include "cuda/Simulation.cuh"
#include "utils/ConfigurationManager.hpp"

namespace sph::benchmark
{

LidDrivenCavity::LidDrivenCavity()
    : ExperimentBase(cuda::Simulation::Parameters::TestCase::LidDrivenCavity)
{
}

auto LidDrivenCavity::createSimulationParameters(const BenchmarkParameters& params,
                                                 const cuda::Simulation::Parameters& simulationParameters,
                                                 BenchmarkResult::SimulationType simulationType)
    -> cuda::Simulation::Parameters
{
    cuda::Simulation::Parameters simulationParams = simulationParameters;
    const auto cavitySize = params.cavitySize;
    simulationParams.domain.min = glm::vec3(-cavitySize / 2, -cavitySize / 2, -cavitySize / 2);
    simulationParams.domain.max = glm::vec3(cavitySize / 2, cavitySize / 2, cavitySize / 2);
    simulationParams.gravity = glm::vec3(0.0F, 0.F, 0.0F);
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
    simulationParams.lidVelocity = params.lidVelocity;
    simulationParams.testCase = cuda::Simulation::Parameters::TestCase::LidDrivenCavity;

    return simulationParams;
}

auto LidDrivenCavity::initializeParticles(const cuda::Simulation::Parameters& simulationParams)
    -> std::vector<glm::vec4>
{
    return initializeParticlesGrid(simulationParams, "Lid-Driven Cavity");
}

auto LidDrivenCavity::createBenchmarkConfig(const BenchmarkParameters& params,
                                            const cuda::Simulation::Parameters& simulationParams) const
    -> BenchmarkResult::SimulationConfig
{
    BenchmarkResult::SimulationConfig config;
    config.cavitySize = params.cavitySize;
    config.lidVelocity = params.lidVelocity;
    config.restDensity = simulationParams.restDensity;
    config.viscosityConstant = simulationParams.viscosityConstant;
    config.domainMin = simulationParams.domain.min;
    config.domainMax = simulationParams.domain.max;
    return config;
}

auto LidDrivenCavity::supportsEnhancedMetrics() const -> bool
{
    return true;
}

void LidDrivenCavity::runSimulation(cuda::Simulation& simulation,
                                    const cuda::Simulation::Parameters& simulationParams,
                                    MetricsCollector& metricsCollector,
                                    uint32_t totalFrames,
                                    uint32_t measureInterval,
                                    float timestep,
                                    Window& window,
                                    BenchmarkVisualizer* visualizer)
{
    // Create configuration for enhanced Cavity metrics
    BenchmarkResult::SimulationConfig config;
    config.restDensity = simulationParams.restDensity;
    config.viscosityConstant = simulationParams.viscosityConstant;
    config.domainMin = simulationParams.domain.min;
    config.domainMax = simulationParams.domain.max;
    // Extract Cavity-specific parameters from simulation parameters
    config.cavitySize = simulationParams.domain.max.x - simulationParams.domain.min.x;  // Domain is cube
    config.lidVelocity = simulationParams.lidVelocity;
    FrameTimeManager timeManager;
    metricsCollector.initialize(simulation, simulationParams.restDensity);
    panda::log::Info("Running Lid Driven Cavity simulation for {} frames, measuring every {} frames",
                     totalFrames,
                     measureInterval);
    SimulationDataGui gui {};
    for (uint32_t frame = 0; frame < totalFrames; ++frame)
    {
        timeManager.update();
        const auto deltaTime = timeManager.getDelta();
        if (window.shouldClose())
        {
            panda::log::Warning("Simulation has been stopped");
            return;
        }
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
            // Use enhanced Cavity metrics collection with performance timing
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

    panda::log::Info("Lid Driven Cavity simulation completed");
}

}
