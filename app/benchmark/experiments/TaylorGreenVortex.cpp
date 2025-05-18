#include "TaylorGreenVortex.hpp"

#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <vector>

#include "ExperimentBase.hpp"
#include "benchmark/MetricsCollector.hpp"
#include "cuda/Simulation.cuh"
#include "utils/ConfigurationManager.hpp"

namespace sph::benchmark
{

TaylorGreenVortex::TaylorGreenVortex()
    : ExperimentBase(cuda::Simulation::Parameters::TestCase::TaylorGreenVortex)
{
}

auto TaylorGreenVortex::createSimulationParameters(const BenchmarkParameters& params,
                                                   const cuda::Simulation::Parameters& simulationParameters,
                                                   BenchmarkResult::SimulationType simulationType)
    -> cuda::Simulation::Parameters
{
    cuda::Simulation::Parameters simulationParams = simulationParameters;
    const auto domainSize = params.domainSize;
    simulationParams.domain.min = glm::vec3(0.0F, 0.0F, 0.0F);
    simulationParams.domain.max = glm::vec3(domainSize, domainSize, domainSize);
    simulationParams.gravity = glm::vec3(0.0F, 0.0F, 0.0F);

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

    // Set test case type
    simulationParams.testCase = cuda::Simulation::Parameters::TestCase::TaylorGreenVortex;

    return simulationParams;
}

auto TaylorGreenVortex::initializeParticles(const cuda::Simulation::Parameters& simulationParams)
    -> std::vector<glm::vec4>
{
    return initializeParticlesGrid(simulationParams, "Taylor-Green Vortex with periodic boundaries");
}

}
