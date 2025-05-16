// TaylorGreenVortex.cpp
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
                                                BenchmarkResult::SimulationType simulationType,
                                                panda::gfx::vulkan::Context& api,
                                                bool visualize,
                                                Window* window)
{
    // Call the base class implementation with visualization enabled
    return ExperimentBase::runBenchmark(params, simulationType, api, visualize, window);
}

cuda::Simulation::Parameters TaylorGreenVortex::createSimulationParameters(
    const BenchmarkParameters& params, BenchmarkResult::SimulationType simulationType)
{
    cuda::Simulation::Parameters simulationParams;

    // Set up the domain for Taylor-Green vortex
    // Using a cubic domain with size 2π as specified in the paper
    float domainSize = params.domainSize;
    simulationParams.domain.min = glm::vec3(0.0f, 0.0f, 0.0f);
    simulationParams.domain.max = glm::vec3(domainSize, domainSize, domainSize);

    // In Taylor-Green vortex, gravity is not typically used
    simulationParams.gravity = glm::vec3(0.0f, 0.0f, 0.0f);

    // Set basic simulation parameters
    simulationParams.restDensity = 1000.0f;
    simulationParams.threadsPerBlock = 256;
    simulationParams.restitution = 0.0f;  // No bounce for this type of flow

    // Set particle size and fluid properties based on simulation type
    if (simulationType == BenchmarkResult::SimulationType::Coarse)
    {
        simulationParams.baseParticleRadius = 0.05F * 1.26F;
        simulationParams.baseParticleMass = 1.2F * 2.F * 1.3F;
        simulationParams.baseSmoothingRadius = 0.22F * 1.26F * 1.1;

        simulationParams.pressureConstant = .5f;
        simulationParams.nearPressureConstant = 0.02f;
        simulationParams.viscosityConstant = 0.002f;
    }
    else if (simulationType == BenchmarkResult::SimulationType::Fine)
    {
        simulationParams.baseParticleRadius = params.fine.particleSize / 2.0f;
        simulationParams.baseParticleMass =
            simulationParams.restDensity * std::pow(simulationParams.baseParticleRadius * 2.0f, 3) * 0.8f;
        simulationParams.baseSmoothingRadius = simulationParams.baseParticleRadius * 4.0f;

        simulationParams.pressureConstant = 0.1f;
        simulationParams.nearPressureConstant = 0.005f;
        simulationParams.viscosityConstant = 0.005f;
    }
    else
    {  // adaptive
        simulationParams.baseParticleRadius =
            (params.adaptive.minParticleSize + params.adaptive.maxParticleSize) / 4.0f;
        simulationParams.baseParticleMass =
            simulationParams.restDensity * std::pow(simulationParams.baseParticleRadius * 2.0f, 3) * 0.8f;
        simulationParams.baseSmoothingRadius = simulationParams.baseParticleRadius * 4.0f;

        simulationParams.pressureConstant = 0.3f;
        simulationParams.nearPressureConstant = 0.008f;
        simulationParams.viscosityConstant = 0.008f;
    }

    // Calculate required viscosity based on Reynolds number
    float characteristicLength = domainSize;
    float characteristicVelocity = 1.0f;  // Unit velocity for Taylor-Green
    float kinematicViscosity = (characteristicVelocity * characteristicLength) / params.reynoldsNumber;
    // Update viscosity constant based on Reynolds number
    simulationParams.viscosityConstant = kinematicViscosity;
    // Limit maximum velocity to avoid instability
    simulationParams.maxVelocity = 5.0f;

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
    panda::log::Info("Last particle at: {}, {}, {}",
                     domainMin.x + (gridSize.x - 1) * actualSpacing.x + offset.x,
                     domainMin.y + (gridSize.y - 1) * actualSpacing.y + offset.y,
                     domainMin.z + (gridSize.z - 1) * actualSpacing.z + offset.z);
}

cuda::refinement::RefinementParameters TaylorGreenVortex::createRefinementParameters(
    const BenchmarkParameters& params,
    BenchmarkResult::SimulationType simulationType,
    const cuda::Simulation::Parameters& simulationParams)
{
    cuda::refinement::RefinementParameters refinementParams;

    // Enable refinement only for adaptive simulation
    refinementParams.enabled = (simulationType == BenchmarkResult::SimulationType::Adaptive);

    // Set refinement parameters
    if (refinementParams.enabled)
    {
        // Calculate mass ratios based on particle sizes
        float minRadius = params.adaptive.minParticleSize / 2.0f;
        float maxRadius = params.adaptive.maxParticleSize / 2.0f;
        float avgRadius = simulationParams.baseParticleRadius;

        // Mass scales with radius^3
        refinementParams.minMassRatio = std::pow(minRadius / avgRadius, 3.0f);
        refinementParams.maxMassRatio = std::pow(maxRadius / avgRadius, 3.0f);

        // For Taylor-Green vortex, vorticity criterion is ideal
        refinementParams.criterionType = "vorticity";

        // Set other refinement parameters
        refinementParams.maxParticleCount = 5000000;
        refinementParams.maxBatchRatio = 0.1f;
        refinementParams.initialCooldown = 100;
        refinementParams.cooldown = 10;

        // Set vorticity thresholds suitable for Taylor-Green vortex
        refinementParams.vorticity.split.minimalVorticityThreshold = 1.0f;
        refinementParams.vorticity.merge.maximalVorticityThreshold = 0.2f;

        // Splitting parameters
        refinementParams.splitting.epsilon = 0.65f;
        refinementParams.splitting.alpha = 0.85f;
        refinementParams.splitting.centerMassRatio = 0.2f;
        refinementParams.splitting.vertexMassRatio = 0.067f;
    }
    else
    {
        // Set a high max particle count for non-adaptive cases
        refinementParams.maxParticleCount = 5000000;
    }

    return refinementParams;
}

}  // namespace sph::benchmark
