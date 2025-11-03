#include <device_atomic_functions.h>

#include <cmath>
#include <cstdint>
#include <cuda/Simulation.cuh>
#include <glm/exponential.hpp>
#include <glm/ext/vector_float4.hpp>

#include "Algorithm.cuh"
#include "glm/geometric.hpp"
#include "kernels/Kernel.cuh"
#include "simulation/SphSimulation.cuh"
#include "utils/Iteration.cuh"
#include "utils/Utils.cuh"

namespace sph::cuda::kernel
{

__device__ void handleCollision(ParticlesData particles, uint32_t id, const Simulation::Parameters& simulationData);
__device__ void handleStandardBoundariesForAxis(ParticlesData particles,
                                                uint32_t id,
                                                const Simulation::Parameters& simulationData,
                                                int axis);

__global__ void handleCollisions(ParticlesData particles, Simulation::Parameters simulationData)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < particles.particleCount)
    {
        handleCollision(particles, idx, simulationData);
    }
}

__global__ void resetGrid(SphSimulation::Grid grid)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < grid.cellStartIndices.size())
    {
        grid.cellStartIndices[idx] = -1;
        grid.cellEndIndices[idx] = -1;
    }
}

__global__ void assignParticlesToCells(ParticlesData particles,
                                       SphSimulation::Grid grid,
                                       Simulation::Parameters simulationData)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < particles.particleCount)
    {
        grid.particleArrayIndices[idx] = idx;
        const auto cellPosition = calculateCellIndex(particles.positions[idx], simulationData, grid);
        const auto cellIndex = flattenCellIndex(cellPosition, grid.gridSize);
        grid.particleGridIndices[idx] = cellIndex;
    }
}

__global__ void calculateCellStartAndEndIndices(SphSimulation::Grid grid, uint32_t particleCount)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particleCount)
    {
        return;
    }

    const auto cellIdx = grid.particleGridIndices[idx];

    if (idx == 0 || grid.particleGridIndices[idx - 1] != cellIdx)
    {
        grid.cellStartIndices[cellIdx] = idx;
    }
    if (idx == grid.particleArrayIndices.size() - 1 || grid.particleGridIndices[idx + 1] != cellIdx)
    {
        grid.cellEndIndices[cellIdx] = idx;
    }
}

__global__ void computeDensities(ParticlesData particles,
                                 SphSimulation::Grid grid,
                                 Simulation::Parameters simulationData)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }

    const auto position = particles.predictedPositions[idx];
    auto density = 0.F;
    auto nearDensity = 0.F;

    forEachNeighbour(position,
                     particles,
                     simulationData,
                     grid,
                     [&](const auto neighbourIdx, const glm::vec4& adjustedPos) {
                         const auto offsetToNeighbour = adjustedPos - position;
                         const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);

                         const auto neighbourSmoothingRadius = particles.smoothingRadiuses[neighbourIdx];
                         const auto radiusSquared = neighbourSmoothingRadius * neighbourSmoothingRadius;

                         if (distanceSquared > radiusSquared)
                         {
                             return;
                         }

                         const auto distance = glm::sqrt(distanceSquared);
                         const auto neighbourMass = particles.masses[neighbourIdx];

                         density += neighbourMass * device::densityKernel(distance, neighbourSmoothingRadius);
                         nearDensity += neighbourMass * device::nearDensityKernel(distance, neighbourSmoothingRadius);
                     });

    particles.densities[idx] = density;
    particles.nearDensities[idx] = nearDensity;
}

__global__ void computePressureForce(ParticlesData particles,
                                     SphSimulation::Grid grid,
                                     Simulation::Parameters simulationData,
                                     float dt)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }
    const auto position = particles.predictedPositions[idx];
    const auto density = particles.densities[idx];
    const auto nearDensity = particles.nearDensities[idx];
    const auto pressure = (density - simulationData.restDensity) * simulationData.pressureConstant;
    const auto nearPressure = nearDensity * simulationData.nearPressureConstant;

    auto pressureForce = glm::vec4 {};

    forEachNeighbour(position,
                     particles,
                     simulationData,
                     grid,
                     [&](const auto neighbourIdx, const glm::vec4& adjustedPos) {
                         if (neighbourIdx == idx)
                         {
                             return;
                         }

                         const auto offsetToNeighbour = adjustedPos - position;
                         const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);
                         const auto neighbourSmoothingRadius = particles.smoothingRadiuses[neighbourIdx];
                         const auto radiusSquared = neighbourSmoothingRadius * neighbourSmoothingRadius;

                         if (distanceSquared > radiusSquared)
                         {
                             return;
                         }
                         const auto densityNeighbour = particles.densities[neighbourIdx];
                         const auto nearDensityNeighbour = particles.nearDensities[neighbourIdx];
                         const auto pressureNeighbour =
                             (densityNeighbour - simulationData.restDensity) * simulationData.pressureConstant;
                         const auto nearPressureNeighbour = nearDensityNeighbour * simulationData.nearPressureConstant;

                         const auto sharedPressure = (pressure + pressureNeighbour) / 2.F;
                         const auto sharedNearPressure = (nearPressure + nearPressureNeighbour) / 2.F;

                         const auto distance = glm::sqrt(distanceSquared);
                         const auto direction =
                             distance > 0.F ? offsetToNeighbour / distance : glm::vec4(0.F, 1.F, 0.F, 0.F);

                         const auto neighbourMass = particles.masses[neighbourIdx];

                         pressureForce += neighbourMass * direction *
                                          device::densityDerivativeKernel(distance, neighbourSmoothingRadius) *
                                          sharedPressure / densityNeighbour;
                         pressureForce += neighbourMass * direction *
                                          device::nearDensityDerivativeKernel(distance, neighbourSmoothingRadius) *
                                          sharedNearPressure / nearDensityNeighbour;
                     });

    const auto particleMass = particles.masses[idx];
    const auto acceleration = (pressureForce / particleMass) / particles.densities[idx];

    particles.velocities[idx] += acceleration * dt;
}

__global__ void computeViscosityForce(ParticlesData particles,
                                      SphSimulation::Grid grid,
                                      Simulation::Parameters simulationData,
                                      float dt)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }

    const auto position = particles.predictedPositions[idx];
    const auto velocity = particles.velocities[idx];

    auto viscosityForce = glm::vec4 {};
    const auto radiusSquared = particles.smoothingRadiuses[idx] * particles.smoothingRadiuses[idx];

    forEachNeighbour(position,
                     particles,
                     simulationData,
                     grid,
                     [&](const auto neighbourIdx, const glm::vec4& adjustedPos) {
                         if (neighbourIdx == idx)
                         {
                             return;
                         }

                         const auto offsetToNeighbour = adjustedPos - position;
                         const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);

                         if (distanceSquared > radiusSquared)
                         {
                             return;
                         }

                         const auto distance = glm::sqrt(distanceSquared);
                         const auto neighbourVelocity = particles.velocities[neighbourIdx];
                         const auto neighbourMass = particles.masses[neighbourIdx];
                         const auto neighbourDensity = particles.densities[neighbourIdx];

                         viscosityForce += neighbourMass * (neighbourVelocity - velocity) / neighbourDensity *
                                           device::viscosityLaplacianKernel(distance, particles.smoothingRadiuses[idx]);
                     });

    const auto particleMass = particles.masses[idx];
    const auto acceleration = simulationData.viscosityConstant * viscosityForce / particleMass;
    particles.velocities[idx] += acceleration * dt;
}

__global__ void integrateMotion(ParticlesData particles, Simulation::Parameters simulationData, float dt)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }

    const auto velocityMagnitude = glm::length(particles.velocities[idx]);
    if (velocityMagnitude > simulationData.maxVelocity)
    {
        particles.velocities[idx] *= simulationData.maxVelocity / velocityMagnitude;
    }
    particles.positions[idx] += particles.velocities[idx] * dt;
}

__device__ void handleCollision(ParticlesData particles, uint32_t id, const Simulation::Parameters& simulationData)
{
    for (int i = 0; i < 3; i++)
    {
        handleStandardBoundariesForAxis(particles, id, simulationData, i);
    }
}

__device__ void handleStandardBoundariesForAxis(ParticlesData particles,
                                                uint32_t id,
                                                const Simulation::Parameters& simulationData,
                                                int axis)
{
    const auto minBoundary = simulationData.domain.min[axis] + particles.radiuses[id];
    const auto maxBoundary = simulationData.domain.max[axis] - particles.radiuses[id];

    if (particles.positions[id][axis] < minBoundary)
    {
        particles.positions[id][axis] = minBoundary;
        particles.velocities[id][axis] = -particles.velocities[id][axis] * simulationData.restitution;
    }

    if (particles.positions[id][axis] > maxBoundary)
    {
        particles.positions[id][axis] = maxBoundary;
        particles.velocities[id][axis] = -particles.velocities[id][axis] * simulationData.restitution;
    }
}

__global__ void computeExternalForces(ParticlesData particles, Simulation::Parameters simulationData, float deltaTime)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }

    particles.velocities[idx] += glm::vec4 {simulationData.gravity, 0.F} * deltaTime;
    particles.predictedPositions[idx] = particles.positions[idx] + particles.velocities[idx] * deltaTime;
}

__global__ void sumAllNeighbors(ParticlesData particles,
                                SphSimulation::Grid grid,
                                Simulation::Parameters simulationData,
                                uint32_t* totalNeighborCount)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }
    const auto position = particles.positions[idx];
    const auto radiusSquared = particles.smoothingRadiuses[idx] * particles.smoothingRadiuses[idx];
    auto count = uint32_t {0};

    forEachNeighbour(position,
                     particles,
                     simulationData,
                     grid,
                     [&](const auto neighbourIdx, const glm::vec4& adjustedPos) {
                         if (idx == neighbourIdx)
                         {
                             return;
                         }

                         const auto offsetToNeighbour = adjustedPos - position;
                         const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);

                         if (distanceSquared <= radiusSquared)
                         {
                             count++;
                         }
                     });

    atomicAdd(totalNeighborCount, count);
}
}
