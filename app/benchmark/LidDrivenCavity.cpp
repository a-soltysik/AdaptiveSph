#include "LidDrivenCavity.hpp"

#include <panda/Logger.h>

namespace sph::benchmark
{

LidDrivenCavity::LidDrivenCavity()
    : ExperimentBase(cuda::Simulation::Parameters::TestCase::LidDrivenCavity)
{
}

BenchmarkResult LidDrivenCavity::runBenchmark(const BenchmarkParameters& params,
                                              const cuda::Simulation::Parameters& simulationParameters,
                                              BenchmarkResult::SimulationType simulationType,
                                              panda::gfx::vulkan::Context& api,
                                              bool visualize,
                                              Window* window)
{
    // Call the base class implementation with visualization enabled
    return ExperimentBase::runBenchmark(params, simulationParameters, simulationType, api, visualize, window);
}

cuda::Simulation::Parameters LidDrivenCavity::createSimulationParameters(
    const BenchmarkParameters& params,
    const cuda::Simulation::Parameters& simulationParameters,
    BenchmarkResult::SimulationType simulationType)
{
    cuda::Simulation::Parameters simulationParams = simulationParameters;

    // Set up the domain for lid-driven cavity
    float cavitySize = params.cavitySize;
    simulationParams.domain.min = glm::vec3(-cavitySize / 2, -cavitySize / 2, -cavitySize / 2);
    simulationParams.domain.max = glm::vec3(cavitySize / 2, cavitySize / 2, cavitySize / 2);

    // Zero gravity for lid-driven cavity
    simulationParams.gravity = glm::vec3(0.0f, 0.f, 0.0f);

    // Set particle size and fluid properties based on simulation type
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

    // Set lid velocity from config
    simulationParams.lidVelocity = params.lidVelocity;

    // Set test case type for custom collision handling
    simulationParams.testCase = cuda::Simulation::Parameters::TestCase::LidDrivenCavity;

    return simulationParams;
}

void LidDrivenCavity::initializeParticles(std::vector<glm::vec4>& particles,
                                          const cuda::Simulation::Parameters& simulationParams)
{
    initializeParticlesGrid(particles, simulationParams, "Lid-Driven Cavity");
}

}  // namespace sph::benchmark
