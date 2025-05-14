// PoiseuilleFlow.cpp
#include "PoiseuilleFlow.hpp"

#include <panda/Logger.h>

namespace sph::benchmark
{

PoiseuilleFlow::PoiseuilleFlow()
    : ExperimentBase(cuda::Simulation::Parameters::TestCase::PoiseuilleFlow)
{
}

BenchmarkResult PoiseuilleFlow::runBenchmark(const BenchmarkParameters& params,
                                             BenchmarkResult::SimulationType simulationType,
                                             panda::gfx::vulkan::Context& api,
                                             bool visualize,
                                             Window* window)
{
    // Call the base class implementation with visualization enabled
    return ExperimentBase::runBenchmark(params, simulationType, api, visualize, window);
}

cuda::Simulation::Parameters PoiseuilleFlow::createSimulationParameters(const BenchmarkParameters& params,
                                                                        BenchmarkResult::SimulationType simulationType)
{
    cuda::Simulation::Parameters simulationParams;

    // Set up the domain for Poiseuille flow
    float halfHeight = params.channelHeight / 2.0f;
    float halfWidth = params.channelWidth / 2.0f;
    float halfLength = params.channelLength / 2.0f;
    simulationParams.domain.min = glm::vec3(-halfLength, -halfHeight, -halfWidth);
    simulationParams.domain.max = glm::vec3(halfLength, halfHeight, halfWidth);

    // Set gravity as a driving force along the channel length (x-axis)
    // For Poiseuille flow, we use gravity as a proxy for pressure gradient
    // Scale by Reynolds number to create appropriate flow conditions
    simulationParams.gravity = glm::vec3(10.F, 0.0f, 0.0f);

    // Set basic simulation parameters
    simulationParams.restDensity = 1000.0f;
    simulationParams.threadsPerBlock = 256;
    simulationParams.restitution = 0.0f;  // No bounce for viscous fluid

    // Set particle size and fluid properties based on simulation type
    if (simulationType == BenchmarkResult::SimulationType::Coarse)
    {
        simulationParams.baseParticleRadius = 0.025F * 1.26F;
        simulationParams.baseParticleMass = 1.2F * 2.F * 0.25F;
        simulationParams.baseSmoothingRadius = 0.22F * 1.26F * 0.55;

        simulationParams.pressureConstant = 0.3f;
        simulationParams.nearPressureConstant = 0.005f;
        simulationParams.viscosityConstant = 0.05f;
    }
    else if (simulationType == BenchmarkResult::SimulationType::Fine)
    {
        simulationParams.baseParticleRadius = 0.025F / 2.F;
        simulationParams.baseParticleMass = 1.2F / 8.F * 0.22;
        simulationParams.baseSmoothingRadius = 0.22F / 2.F * 0.65;

        simulationParams.pressureConstant = 0.03f;
        simulationParams.nearPressureConstant = 0.002f;
        simulationParams.viscosityConstant = 0.0001f;
    }
    else
    {  // adaptive
        simulationParams.baseParticleRadius = 0.025F;
        simulationParams.baseParticleMass = 1.2F * 0.25;
        simulationParams.baseSmoothingRadius = 0.22F * 0.65;

        simulationParams.pressureConstant = .5f;
        simulationParams.nearPressureConstant = 0.005f;
        simulationParams.viscosityConstant = 0.001f;
    }

    // Tune viscosity based on Reynolds number
    // For Poiseuille flow: Re = (mean velocity * channel height) / kinematic viscosity
    // Mean velocity is determined by the pressure gradient (gravity in our case)
    //float kinematicViscosity = (params.channelHeight * params.channelHeight) / params.reynoldsNumber;
    //simulationParams.viscosityConstant = kinematicViscosity * 10.0f;  // Scale factor for SPH formulation

    // Limit maximum velocity to avoid instability
    simulationParams.maxVelocity = 10.0f;

    // Set test case type for custom collision handling
    simulationParams.testCase = cuda::Simulation::Parameters::TestCase::PoiseuilleFlow;

    return simulationParams;
}

void PoiseuilleFlow::initializeParticles(std::vector<glm::vec4>& particles,
                                         const cuda::Simulation::Parameters& simulationParams)
{
    // Clear any existing particles
    particles.clear();

    // Calculate the domain size
    const glm::vec3 domainMin = simulationParams.domain.min;
    const glm::vec3 domainMax = simulationParams.domain.max;
    const glm::vec3 domainSize = domainMax - domainMin;

    panda::log::Info("Domain size: {}x{}x{}", domainSize.x, domainSize.y, domainSize.z);

    // Calculate particle spacing based on particle diameter (2 * radius)
    float particleSpacing = simulationParams.baseParticleRadius * 2.0f * 1.1F;  // Slight overlap for stability

    // Calculate grid size
    glm::uvec3 gridSize;
    gridSize.x = static_cast<uint32_t>(domainSize.x / particleSpacing);
    gridSize.y = static_cast<uint32_t>(domainSize.y / particleSpacing);
    gridSize.z = static_cast<uint32_t>(domainSize.z / particleSpacing);

    // Ensure at least one particle in each dimension
    gridSize.x = std::max(gridSize.x, 1u);
    gridSize.y = std::max(gridSize.y, 1u);
    gridSize.z = std::max(gridSize.z, 1u);

    panda::log::Info("Creating Poiseuille flow with grid size: {}x{}x{}", gridSize.x, gridSize.y, gridSize.z);

    // Calculate actual spacing to evenly distribute particles
    glm::vec3 actualSpacing;
    actualSpacing.x = domainSize.x / (gridSize.x + 1);
    actualSpacing.y = domainSize.y / (gridSize.y + 1);
    actualSpacing.z = domainSize.z / (gridSize.z + 1);

    // Create particles throughout the domain, but keep a small gap near the walls
    // This helps with the no-slip boundary condition
    const float wallGap = simulationParams.baseParticleRadius * 0.5f;
    const glm::vec3 startPos = domainMin + glm::vec3(actualSpacing.x, wallGap, wallGap);
    for (uint32_t i = 0; i < gridSize.x; i++)
    {
        for (uint32_t j = 0; j < gridSize.y; j++)
        {
            for (uint32_t k = 0; k < gridSize.z; k++)
            {
                const auto position = startPos + glm::vec3(actualSpacing.x * static_cast<float>(i),
                                                           actualSpacing.y * static_cast<float>(j),
                                                           actualSpacing.z * static_cast<float>(k));
                // Check if the particle is within the proper domain bounds
                if (position.y >= domainMin.y + wallGap && position.y <= domainMax.y - wallGap &&
                    position.z >= domainMin.z + wallGap && position.z <= domainMax.z - wallGap)
                {
                    particles.emplace_back(position, 0.0f);
                }
            }
        }
    }

    panda::log::Info("Created {} particles for Poiseuille flow", particles.size());
}

cuda::refinement::RefinementParameters PoiseuilleFlow::createRefinementParameters(
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
        float minRadius = params.adaptive.minParticleSize;
        float maxRadius = params.adaptive.maxParticleSize;
        float avgRadius = simulationParams.baseParticleRadius;

        // Mass scales with radius^3
        //refinementParams.minMassRatio = std::pow(minRadius / avgRadius, 3.0f);
        //refinementParams.maxMassRatio = std::pow(maxRadius / avgRadius, 3.0f);

        // For Poiseuille flow, velocity gradient is an ideal criterion
        refinementParams.criterionType = "velocity";

        // Set other refinement parameters
        refinementParams.maxParticleCount = 10000000;
        refinementParams.maxBatchRatio = 0.1f;
        refinementParams.initialCooldown = 100;
        refinementParams.cooldown = 10;

        // Set velocity thresholds for a Poiseuille flow
        // Higher refinement near walls where velocity gradients are highest
        refinementParams.velocity.split.minimalSpeedThreshold = 1.5f;
        refinementParams.velocity.merge.maximalSpeedThreshold = 0.3f;

        // Splitting parameters
        refinementParams.splitting.epsilon = 0.65f;
        refinementParams.splitting.alpha = 0.85f;
        refinementParams.splitting.centerMassRatio = 0.2f;
        refinementParams.splitting.vertexMassRatio = 0.067f;
    }
    else
    {
        // Set a high max particle count for non-adaptive cases
        refinementParams.maxParticleCount = 10000000;
    }

    return refinementParams;
}

}  // namespace sph::benchmark
