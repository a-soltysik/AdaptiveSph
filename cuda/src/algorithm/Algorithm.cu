#include <device_atomic_functions.h>

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

    if (idx == particleCount - 1)
    {
        grid.cellEndIndices[cellIdx] = idx;
        return;
    }
    if (idx == 0)
    {
        grid.cellStartIndices[cellIdx] = idx;
        return;
    }

    const auto cellIdxNext = grid.particleGridIndices[idx + 1];
    if (cellIdx != cellIdxNext)
    {
        grid.cellStartIndices[cellIdxNext] = idx + 1;
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

    const auto position = particles.positions[idx];
    const auto smoothingRadius = particles.smoothingRadiuses[idx];
    auto density = float {};

    forEachNeighbour(position,
                     particles.positions,
                     simulationData.domain,
                     grid,
                     device::constant::wendlandRangeRatio * smoothingRadius,
                     [position, &particles, &density](const auto neighbourIdx, const auto neighborPosition) {
                         const auto offsetToNeighbour = neighborPosition - position;
                         const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);

                         const auto neighbourSmoothingRadius = particles.smoothingRadiuses[neighbourIdx];
                         const auto radiusSquared = (device::constant::wendlandRangeRatio * neighbourSmoothingRadius) *
                                                    (device::constant::wendlandRangeRatio * neighbourSmoothingRadius);

                         if (distanceSquared > radiusSquared)
                         {
                             return;
                         }

                         const auto distance = glm::sqrt(distanceSquared);
                         const auto neighbourMass = particles.masses[neighbourIdx];

                         density += neighbourMass * device::wendlandKernel(distance, neighbourSmoothingRadius);
                     });

    particles.densities[idx] = density;
}

__device__ auto computeTaitPressure(float density, float restDensity, float speedOfSound) -> float
{
    static constexpr auto gamma = 7.F;
    const auto B = restDensity * speedOfSound * speedOfSound / gamma;
    const auto densityRatio = density / restDensity;
    return B * (powf(densityRatio, gamma) - 1.F);
}

__global__ void computePressureAccelerations(ParticlesData particles,
                                             SphSimulation::Grid grid,
                                             Simulation::Parameters simulationData)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }
    const auto position = particles.positions[idx];
    const auto density = particles.densities[idx];
    const auto smoothingRadius = particles.smoothingRadiuses[idx];
    const auto pressure = computeTaitPressure(density, simulationData.restDensity, simulationData.speedOfSound);

    auto acceleration = glm::vec4 {};

    forEachNeighbour(
        position,
        particles.positions,
        simulationData.domain,
        grid,
        device::constant::wendlandRangeRatio * smoothingRadius,
        [position, &particles, &simulationData, pressure, density, &acceleration, idx](const auto neighbourIdx,
                                                                                       const auto neighborPosition) {
            if (neighbourIdx == idx)
            {
                return;
            }

            const auto offsetToNeighbour = position - neighborPosition;
            const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);
            const auto neighbourSmoothingRadius = particles.smoothingRadiuses[neighbourIdx];
            const auto radiusSquared = (device::constant::wendlandRangeRatio * neighbourSmoothingRadius) *
                                       (device::constant::wendlandRangeRatio * neighbourSmoothingRadius);

            if (distanceSquared > radiusSquared)
            {
                return;
            }
            const auto densityNeighbor = particles.densities[neighbourIdx];
            const auto distance = glm::sqrt(distanceSquared);
            const auto direction = distance > 0.F ? offsetToNeighbour / distance : glm::vec4(0.F, 1.F, 0.F, 0.F);
            const auto pressureNeighbor =
                computeTaitPressure(densityNeighbor, simulationData.restDensity, simulationData.speedOfSound);

            const auto neighbourMass = particles.masses[neighbourIdx];

            const auto pressureTerm =
                pressure / (density * density) + (pressureNeighbor / (densityNeighbor * densityNeighbor));
            acceleration -= neighbourMass * direction *
                            device::wendlandDerivativeKernel(distance, neighbourSmoothingRadius) * pressureTerm;
        });

    particles.accelerations[idx] += acceleration;
}

__global__ void computeViscosityAccelerations(ParticlesData particles,
                                              SphSimulation::Grid grid,
                                              Simulation::Parameters simulationData)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }

    const auto position = particles.positions[idx];
    const auto velocity = particles.velocities[idx];
    const auto smoothingRadius = particles.smoothingRadiuses[idx];
    const auto density = particles.densities[idx];

    auto acceleration = glm::vec4 {};

    static constexpr auto epsilon = 0.01F;

    forEachNeighbour(
        position,
        particles.positions,
        simulationData.domain,
        grid,
        device::constant::wendlandRangeRatio * smoothingRadius,
        [position, velocity, &particles, &acceleration, &simulationData, density, smoothingRadius, idx](
            const auto neighborIdx,
            const auto neighborPosition) {
            if (neighborIdx == idx)
            {
                return;
            }

            const auto offsetToNeighbour = position - neighborPosition;
            const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);
            const auto neighborSmoothingRadius = particles.smoothingRadiuses[neighborIdx];
            const auto radiusSquared = (device::constant::wendlandRangeRatio * neighborSmoothingRadius) *
                                       (device::constant::wendlandRangeRatio * neighborSmoothingRadius);

            if (distanceSquared > radiusSquared)
            {
                return;
            }

            const auto neighborVelocity = particles.velocities[neighborIdx];
            const auto velocityDifference = velocity - neighborVelocity;
            const auto compression = glm::dot(velocityDifference, offsetToNeighbour);
            if (compression >= 0.F)
            {
                return;
            }

            const auto distance = glm::sqrt(distanceSquared);
            const auto neighborMass = particles.masses[neighborIdx];
            const auto neighborDensity = particles.densities[neighborIdx];
            const auto nu = 2.F * simulationData.viscosityConstant * smoothingRadius * simulationData.speedOfSound /
                            (density + neighborDensity);

            const auto pi = -nu * compression / (distanceSquared + epsilon * smoothingRadius * smoothingRadius);
            const auto direction = distance > 0.F ? offsetToNeighbour / distance : glm::vec4(0.F, 1.F, 0.F, 0.F);

            acceleration -=
                neighborMass * pi * device::wendlandDerivativeKernel(distance, neighborSmoothingRadius) * direction;
        });

    particles.accelerations[idx] += acceleration;
}

__global__ void computeSurfaceTensionAccelerations(ParticlesData particles,
                                                   SphSimulation::Grid grid,
                                                   Simulation::Parameters simulationData)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }

    const auto position = particles.positions[idx];
    const auto smoothingRadius = particles.smoothingRadiuses[idx];
    const auto mass = particles.masses[idx];

    auto acceleration = glm::vec4 {};

    forEachNeighbour(
        position,
        particles.positions,
        simulationData.domain,
        grid,
        device::constant::wendlandRangeRatio * smoothingRadius,
        [position, &particles, &acceleration, idx](const auto neighborIdx, const glm::vec4& adjustedPos) {
            if (neighborIdx == idx)
            {
                return;
            }

            const auto offsetToNeighbour = adjustedPos - position;
            const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);
            const auto neighborSmoothingRadius = particles.smoothingRadiuses[neighborIdx];
            const auto radiusSquared = (device::constant::wendlandRangeRatio * neighborSmoothingRadius) *
                                       (device::constant::wendlandRangeRatio * neighborSmoothingRadius);

            if (distanceSquared > radiusSquared)
            {
                return;
            }

            const auto distance = glm::sqrt(distanceSquared);
            const auto neighborMass = particles.masses[neighborIdx];
            const auto direction = distance > 0.F ? offsetToNeighbour / distance : glm::vec4(0.F, 1.F, 0.F, 0.F);

            acceleration -= neighborMass * device::wendlandKernel(distance, neighborSmoothingRadius) * direction;
        });

    particles.accelerations[idx] += (simulationData.surfaceTensionConstant / mass) * acceleration;
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

__global__ void computeExternalAccelerations(ParticlesData particles, Simulation::Parameters simulationData)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }

    particles.accelerations[idx] = glm::vec4 {simulationData.gravity, 0.F};
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
    const auto smoothingRadius = particles.smoothingRadiuses[idx];
    auto count = uint32_t {0};

    forEachNeighbour(position,
                     particles.positions,
                     simulationData.domain,
                     grid,
                     device::constant::wendlandRangeRatio * smoothingRadius,
                     [position, &particles, &count, idx](const auto neighborIdx, const glm::vec4& adjustedPos) {
                         if (idx == neighborIdx)
                         {
                             return;
                         }

                         const auto offsetToNeighbour = adjustedPos - position;
                         const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);
                         const auto neighborSmoothingRadius = particles.smoothingRadiuses[neighborIdx];
                         const auto radiusSquared = (device::constant::wendlandRangeRatio * neighborSmoothingRadius) *
                                                    (device::constant::wendlandRangeRatio * neighborSmoothingRadius);

                         if (distanceSquared <= radiusSquared)
                         {
                             count++;
                         }
                     });

    atomicAdd(totalNeighborCount, count);
}

__global__ void halfKickVelocities(ParticlesData particles, Simulation::Parameters simulationData, float halfDt)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }
    particles.velocities[idx] += particles.accelerations[idx] * halfDt;

    const auto velocityMagnitude = glm::length(particles.velocities[idx]);
    if (velocityMagnitude > simulationData.maxVelocity)
    {
        particles.velocities[idx] *= simulationData.maxVelocity / velocityMagnitude;
    }
}

__global__ void updatePositions(ParticlesData particles, float dt)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }
    particles.positions[idx] += particles.velocities[idx] * dt;
}

}
