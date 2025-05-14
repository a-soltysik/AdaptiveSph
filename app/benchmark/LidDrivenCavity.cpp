// LidDrivenCavity.cpp
#include "LidDrivenCavity.hpp"

#include <panda/Logger.h>

namespace sph::benchmark
{

LidDrivenCavity::LidDrivenCavity()
    : ExperimentBase(cuda::Simulation::Parameters::TestCase::LidDrivenCavity)
{
}

BenchmarkResult LidDrivenCavity::runBenchmark(const BenchmarkParameters& params,
                                              BenchmarkResult::SimulationType simulationType,
                                              panda::gfx::vulkan::Context& api,
                                              bool visualize,
                                              Window* window)
{
    // Call the base class implementation with visualization enabled
    return ExperimentBase::runBenchmark(params, simulationType, api, visualize, window);
}

cuda::Simulation::Parameters LidDrivenCavity::createSimulationParameters(const BenchmarkParameters& params,
                                                                         BenchmarkResult::SimulationType simulationType)
{
    cuda::Simulation::Parameters simulationParams;

    // Set up the domain for lid-driven cavity
    float cavitySize = params.cavitySize;
    simulationParams.domain.min = glm::vec3(-cavitySize / 2, -cavitySize / 2, -cavitySize / 2);
    simulationParams.domain.max = glm::vec3(cavitySize / 2, cavitySize / 2, cavitySize / 2);

    // Zero gravity for lid-driven cavity
    simulationParams.gravity = glm::vec3(0.0f, 0.f, 0.0f);

    // Set basic simulation parameters
    simulationParams.restDensity = 1000.0f;
    simulationParams.threadsPerBlock = 256;
    simulationParams.restitution = 0.5f;  // No bounce for viscous fluid

    // Set particle size based on simulation type
    if (simulationType == BenchmarkResult::SimulationType::Coarse)
    {
        simulationParams.baseParticleRadius = 0.025F * 1.26F;
        simulationParams.baseParticleMass = 1.2F * 2.F * 0.4F;
        simulationParams.baseSmoothingRadius = 0.22F * 1.26F * 0.65;

        simulationParams.pressureConstant = .5f;
        simulationParams.nearPressureConstant = .1f;
        simulationParams.viscosityConstant = 0.001;  // Scale factor depends on implementation
    }
    else if (simulationType == BenchmarkResult::SimulationType::Fine)
    {
        simulationParams.baseParticleRadius = 0.025F / 2.F;
        simulationParams.baseParticleMass = 1.2F / 8.F * 0.35;
        simulationParams.baseSmoothingRadius = 0.22F / 2.F * 0.65;

        simulationParams.pressureConstant = .1;
        simulationParams.nearPressureConstant = 0.001f;
        simulationParams.viscosityConstant = 0.0001;  // Scale factor depends on implementation
    }
    else
    {  // adaptive
        simulationParams.baseParticleRadius = 0.025F;
        simulationParams.baseParticleMass = 1.2F * 0.4;
        simulationParams.baseSmoothingRadius = 0.22F * 0.65;

        simulationParams.pressureConstant = .1f;
        simulationParams.nearPressureConstant = 0.001f;
        simulationParams.viscosityConstant = 0.0001;  // Scale factor depends on implementation
    }

    // Set viscosity based on Reynolds number
    // For lid-driven cavity: Re = (lid velocity * cavity size) / kinematic viscosity
    float lidVelocity = 5.0f;  // Unit velocity for simplicity
    simulationParams.lidVelocity = lidVelocity;

    // Calculate required viscosity for the desired Reynolds number
    // SPH viscosity parameter is related to kinematic viscosity
    float kinematicViscosity = (lidVelocity * cavitySize) / params.reynoldsNumber;

    // This is a simplified model - in reality, the relationship between SPH viscosity constant
    // and physical kinematic viscosity is more complex and dependent on your SPH implementation
    //simulationParams.viscosityConstant = kinematicViscosity * 10.0f;  // Scale factor depends on implementation

    // Limit maximum velocity to avoid instability
    simulationParams.maxVelocity = lidVelocity * 2.0f;

    // Set test case type for custom collision handling
    simulationParams.testCase = cuda::Simulation::Parameters::TestCase::LidDrivenCavity;

    return simulationParams;
}

void LidDrivenCavity::initializeParticles(std::vector<glm::vec4>& particles,
                                          const cuda::Simulation::Parameters& simulationParams)
{
    // Clear any existing particles
    particles.clear();

    // Calculate the domain size
    const glm::vec3 domainMin = simulationParams.domain.min;
    const glm::vec3 domainMax = simulationParams.domain.max;
    const glm::vec3 domainSize = domainMax - domainMin;

    // Calculate particle spacing based on particle diameter (2 * radius)
    float particleSpacing = simulationParams.baseParticleRadius * 2.0f * 1.3F;  // Slight overlap for stability

    // Calculate grid size
    glm::uvec3 gridSize;
    gridSize.x = static_cast<uint32_t>(domainSize.x / particleSpacing);
    gridSize.y = static_cast<uint32_t>(domainSize.y / particleSpacing);
    gridSize.z = static_cast<uint32_t>(domainSize.z / particleSpacing);

    // Ensure at least one particle in each dimension
    gridSize.x = std::max(gridSize.x, 1u);
    gridSize.y = std::max(gridSize.y, 1u);
    gridSize.z = std::max(gridSize.z, 1u);

    panda::log::Info("Creating lid-driven cavity with grid size: {}x{}x{}", gridSize.x, gridSize.y, gridSize.z);

    // Calculate actual spacing to evenly distribute particles
    glm::vec3 actualSpacing;
    actualSpacing.x = domainSize.x / (gridSize.x + 1);
    actualSpacing.y = domainSize.y / (gridSize.y + 1);
    actualSpacing.z = domainSize.z / (gridSize.z + 1);

    // Create particles throughout the domain
    const glm::vec3 startPos = domainMin + actualSpacing;
    for (uint32_t i = 0; i < gridSize.x; i++)
    {
        for (uint32_t j = 0; j < gridSize.y; j++)
        {
            for (uint32_t k = 0; k < gridSize.z; k++)
            {
                const auto position = startPos + glm::vec3(actualSpacing.x * static_cast<float>(i),
                                                           actualSpacing.y * static_cast<float>(j),
                                                           actualSpacing.z * static_cast<float>(k));

                particles.emplace_back(position, 0.0f);
            }
        }
    }

    panda::log::Info("Created {} particles for lid-driven cavity", particles.size());
}

cuda::refinement::RefinementParameters LidDrivenCavity::createRefinementParameters(
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
        refinementParams.minMassRatio = std::pow(minRadius / avgRadius, 3.0f);
        refinementParams.maxMassRatio = std::pow(maxRadius / avgRadius, 3.0f);

        // Use curvature criterion for lid-driven cavity (good for vortices)
        refinementParams.criterionType = "interface";

        // Set other refinement parameters to reasonable values
        refinementParams.maxParticleCount = 100000;
        refinementParams.maxBatchRatio = 0.1f;
        refinementParams.initialCooldown = 1000;
        refinementParams.cooldown = 10;

        // Set curvature thresholds
        refinementParams.curvature.split.minimalCurvatureThreshold = 15.0f;
        refinementParams.curvature.merge.maximalCurvatureThreshold = 5.0f;

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

I have working normal and adaptive refinement SPH simulation. However now I need to perform tests on the simulation in order to provide the analysis of my project.

I need a benchamrking test framework. In the framework I will need to compare 3 simulation:
1. Simulation with big particles (the size should be configurable), normal SPH
2. Simulation with small particles (the size should be configurable), normal SPH
3. Simulation with medium particles, but using adaptyive SPH

For each type of simulation I need to have adjustible pressure constant, near pressure constant, baseSmothingRadius, baseParticleRadius, baseParticleMass and Reynolds number (the refinement parameters can be used from refinementr section in json file, because there will be only one refinement case in each test)

The bencharking framework will in the end include several types of experiments, however, for now I need lid-driven cavity. Please propose extensible framework, so that I will be able to add new experimets easily.

You can modify whole codebase. For example I know thet for lid-driven cavity we will need top wall to be "moving". So for example we can just instead of bouncing particle when it hits a top wall, it will bounced parallel to the wall with velocity V (this way I think we can simulate moving top wall, please tell me if I am wrong). To do this I think you will need communicate the type of experiment from App to SphSimulation and change the impleemntation for handleCollision (but only for this experiment)

Please, let your code be clean. I mean try not to cerate big function and if you need to impleemnt new feature, meybe it will be better to create new class.

Provide full implementation of code you will create. I mean, do not ever create empty function that should be implemented by me. All functions should be implemented.

The test frameowrk should also capture data from the simulation into json or csv file.

Also the framework should collect data like:
* l2 density error norm
* pressure field smoothness
* mass conservation error
* time to simulate a frame
* particle clustering index (Detects unphysical particle clustering, a common instability in SPH)
