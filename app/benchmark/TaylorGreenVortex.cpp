#include "TaylorGreenVortex.hpp"

#include <panda/Logger.h>

#include <glm/gtc/constants.hpp>

namespace sph::benchmark
{

TaylorGreenVortex::TaylorGreenVortex()
    : ExperimentBase(cuda::Simulation::Parameters::TestCase::TaylorGreenVortex)
{
}

BenchmarkResult TaylorGreenVortex::runBenchmark(const BenchmarkParameters& params,
                                                const cuda::Simulation::Parameters& simulationParameters,
                                                BenchmarkResult::SimulationType simulationType,
                                                panda::gfx::vulkan::Context& api,
                                                bool visualize,
                                                Window* window)
{
    // Call the base class implementation with visualization enabled
    return ExperimentBase::runBenchmark(params, simulationParameters, simulationType, api, visualize, window);
}

cuda::Simulation::Parameters TaylorGreenVortex::createSimulationParameters(
    const BenchmarkParameters& params,
    const cuda::Simulation::Parameters& simulationParameters,
    BenchmarkResult::SimulationType simulationType)
{
    cuda::Simulation::Parameters simulationParams = simulationParameters;

    // Set up the domain for Taylor-Green vortex
    // Using a cubic domain with size 2π as specified in the paper
    float domainSize = params.domainSize;
    simulationParams.domain.min = glm::vec3(0.0f, 0.0f, 0.0f);
    simulationParams.domain.max = glm::vec3(domainSize, domainSize, domainSize);

    // In Taylor-Green vortex, gravity is not typically used
    simulationParams.gravity = glm::vec3(0.0f, 0.0f, 0.0f);

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

    // Set test case type
    simulationParams.testCase = cuda::Simulation::Parameters::TestCase::TaylorGreenVortex;

    return simulationParams;
}

void TaylorGreenVortex::initializeParticles(std::vector<glm::vec4>& particles,
                                            const cuda::Simulation::Parameters& simulationParams)
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
    // For periodic domains, we use one fewer particle than would fit exactly
    // This prevents duplicate particles at domain boundaries
    glm::uvec3 gridSize;
    gridSize.x = static_cast<uint32_t>(std::floor(domainSize.x / particleSpacing));
    gridSize.y = static_cast<uint32_t>(std::floor(domainSize.y / particleSpacing));
    gridSize.z = static_cast<uint32_t>(std::floor(domainSize.z / particleSpacing));

    panda::log::Info("Creating Taylor-Green vortex with grid size: {}x{}x{}", gridSize.x, gridSize.y, gridSize.z);

    // Calculate actual spacing to distribute particles evenly
    glm::vec3 actualSpacing;
    actualSpacing.x = domainSize.x / static_cast<float>(gridSize.x);
    actualSpacing.y = domainSize.y / static_cast<float>(gridSize.y);
    actualSpacing.z = domainSize.z / static_cast<float>(gridSize.z);

    // Calculate offset to center particles within the domain
    // This ensures particles are not placed directly on domain boundaries
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

    panda::log::Info("Created {} particles for Taylor-Green vortex with periodic boundaries", particles.size());
    panda::log::Info("Particle spacing: {}, {}, {}", actualSpacing.x, actualSpacing.y, actualSpacing.z);
    panda::log::Info("First particle at: {}, {}, {}",
                     domainMin.x + offset.x,
                     domainMin.y + offset.y,
                     domainMin.z + offset.z);
}

}  // namespace sph::benchmark
