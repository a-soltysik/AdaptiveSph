#include "AlgorithmENS.cuh"
#include "glm/ext/vector_float3.hpp"
#include "kernels/Kernel.cuh"
#include "utils/IterationENS.cuh"

namespace sph::cuda::kernel
{

__global__ void computeParticleRectangles(ParticlesData particles,
                                          Rectangle3D* rectangles,
                                          SphSimulation::Grid grid,
                                          Simulation::Parameters simulationData)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= particles.particleCount)
    {
        return;
    }

    const auto position = particles.predictedPositions[idx];
    const auto radius = particles.smoothingRadiuses[idx];
    const auto relativePosition = glm::vec3 {position} - simulationData.domain.min;
    const auto min = glm::max(glm::ivec3 {(relativePosition - radius) / grid.cellSize}, glm::ivec3 {0, 0, 0});
    const auto max = glm::min(glm::ivec3 {(relativePosition + radius) / grid.cellSize}, grid.gridSize - 1);

    rectangles[idx] = {.min = min, .max = max};
}

__global__ void computeDensitiesENS(ParticlesData particles,
                                    const Rectangle3D* particleRectangles,
                                    SphSimulation::Grid grid,
                                    Simulation::Parameters simulationData,
                                    int32_t particlesPerBatch)
{
    extern __shared__ char sharedMemory[];
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= particles.particleCount)
    {
        return;
    }

    const auto particleRect = particleRectangles[idx];

    auto density = 0.F;
    auto nearDensity = 0.F;

    forEachNeighbourENS(
        particleRect,
        particles,
        grid,
        simulationData,
        sharedMemory,
        particlesPerBatch,
        [&particles, &density, &nearDensity, idx](const auto neighborIdx, const glm::vec4& neighborPosition) {
            const auto position = particles.predictedPositions[idx];
            const auto offsetToNeighbour = neighborPosition - position;
            const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);

            const auto neighbourSmoothingRadius = particles.smoothingRadiuses[neighborIdx];
            const auto radiusSquared = neighbourSmoothingRadius * neighbourSmoothingRadius;
            if (distanceSquared > radiusSquared)
            {
                return;
            }

            const auto distance = glm::sqrt(distanceSquared);
            const auto neighbourMass = particles.masses[neighborIdx];

            density += neighbourMass * device::densityKernel(distance, neighbourSmoothingRadius);
            nearDensity += neighbourMass * device::nearDensityKernel(distance, neighbourSmoothingRadius);
        });
    particles.densities[idx] = density;
    particles.nearDensities[idx] = nearDensity;
}

__global__ void computePressureForceENS(ParticlesData particles,
                                        const Rectangle3D* particleRectangles,
                                        SphSimulation::Grid grid,
                                        Simulation::Parameters simulationData,
                                        float dt,
                                        int32_t particlesPerBatch)
{
    extern __shared__ char sharedMemory[];
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= particles.particleCount)
    {
        return;
    }

    const auto particleRect = particleRectangles[idx];
    const auto density = particles.densities[idx];
    const auto nearDensity = particles.nearDensities[idx];
    const auto pressure = (density - simulationData.restDensity) * simulationData.pressureConstant;
    const auto nearPressure = nearDensity * simulationData.nearPressureConstant;

    auto pressureForce = glm::vec4 {};

    forEachNeighbourENS(
        particleRect,
        particles,
        grid,
        simulationData,
        sharedMemory,
        particlesPerBatch,
        [&](const auto neighborIdx, const glm::vec4& neighborPosition) {
            if (neighborIdx == idx)
            {
                return;
            }
            const auto position = particles.predictedPositions[idx];
            const auto offsetToNeighbour = neighborPosition - position;
            const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);
            const auto neighbourSmoothingRadius = particles.smoothingRadiuses[neighborIdx];
            const auto radiusSquared = neighbourSmoothingRadius * neighbourSmoothingRadius;

            if (distanceSquared > radiusSquared)
            {
                return;
            }

            const auto densityNeighbour = particles.densities[neighborIdx];
            const auto nearDensityNeighbour = particles.nearDensities[neighborIdx];
            const auto pressureNeighbour =
                (densityNeighbour - simulationData.restDensity) * simulationData.pressureConstant;
            const auto nearPressureNeighbour = nearDensityNeighbour * simulationData.nearPressureConstant;

            const auto sharedPressure = (pressure + pressureNeighbour) / 2.F;
            const auto sharedNearPressure = (nearPressure + nearPressureNeighbour) / 2.F;

            const auto distance = glm::sqrt(distanceSquared);
            const auto direction = distance > 0.F ? offsetToNeighbour / distance : glm::vec4(0.F, 1.F, 0.F, 0.F);

            const auto neighbourMass = particles.masses[neighborIdx];

            pressureForce += neighbourMass * direction *
                             device::densityDerivativeKernel(distance, neighbourSmoothingRadius) * sharedPressure /
                             densityNeighbour;
            pressureForce += neighbourMass * direction *
                             device::nearDensityDerivativeKernel(distance, neighbourSmoothingRadius) *
                             sharedNearPressure / nearDensityNeighbour;
        });

    const auto particleMass = particles.masses[idx];
    const auto acceleration = (pressureForce / particleMass) / particles.densities[idx];

    particles.velocities[idx] += acceleration * dt;
}

__global__ void computeViscosityForceENS(ParticlesData particles,
                                         const Rectangle3D* particleRectangles,
                                         SphSimulation::Grid grid,
                                         Simulation::Parameters simulationData,
                                         float dt,
                                         int32_t particlesPerBatch)
{
    extern __shared__ char sharedMemory[];
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= particles.particleCount)
    {
        return;
    }

    const auto particleRect = particleRectangles[idx];
    auto viscosityForce = glm::vec4 {};

    forEachNeighbourENS(
        particleRect,
        particles,
        grid,
        simulationData,
        sharedMemory,
        particlesPerBatch,
        [&particles, &viscosityForce, idx](const auto neighborIdx, const glm::vec4& neighborPosition) {
            if (neighborIdx == idx)
            {
                return;
            }
            const auto position = particles.predictedPositions[idx];

            const auto offsetToNeighbour = neighborPosition - position;
            const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);
            const auto radiusSquared = particles.smoothingRadiuses[idx] * particles.smoothingRadiuses[idx];

            if (distanceSquared > radiusSquared)
            {
                return;
            }

            const auto distance = glm::sqrt(distanceSquared);
            const auto neighbourVelocity = particles.velocities[neighborIdx];
            const auto neighbourMass = particles.masses[neighborIdx];
            const auto neighbourDensity = particles.densities[neighborIdx];

            const auto velocity = particles.velocities[idx];
            viscosityForce += neighbourMass * (neighbourVelocity - velocity) / neighbourDensity *
                              device::viscosityLaplacianKernel(distance, particles.smoothingRadiuses[idx]);
        });

    const auto particleMass = particles.masses[idx];
    const auto acceleration = simulationData.viscosityConstant * viscosityForce / particleMass;
    particles.velocities[idx] += acceleration * dt;
}

}
