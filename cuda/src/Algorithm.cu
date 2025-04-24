#include <cuda/std/__iterator/distance.h>

#include <cstdint>
#include <cuda/Simulation.cuh>
#include <cuda/std/array>
#include <glm/common.hpp>
#include <glm/ext/vector_uint3.hpp>
#include <thrust/system/detail/generic/distance.inl>

#include "Algorithm.cuh"
#include "Span.cuh"
#include "SphSimulation.cuh"
#include "device/Kernel.cuh"
#include "glm/ext/vector_float3.hpp"
#include "glm/geometric.hpp"

namespace sph::cuda::kernel
{

__constant__ int3 offsets[27] = {
    {0,  0,  0 },
    {0,  0,  -1},
    {0,  0,  1 },
    {0,  -1, 0 },
    {0,  -1, -1},
    {0,  -1, 1 },
    {0,  1,  0 },
    {0,  1,  -1},
    {0,  1,  1 },
    {-1, 0,  0 },
    {-1, 0,  -1},
    {-1, 0,  1 },
    {-1, -1, 0 },
    {-1, -1, -1},
    {-1, -1, 1 },
    {-1, 1,  0 },
    {-1, 1,  -1},
    {-1, 1,  1 },
    {1,  0,  0 },
    {1,  0,  -1},
    {1,  0,  1 },
    {1,  -1, 0 },
    {1,  -1, -1},
    {1,  -1, 1 },
    {1,  1,  0 },
    {1,  1,  -1},
    {1,  1,  1 }
};

__device__ void handleCollision(ParticlesData particles, uint32_t id, const Simulation::Parameters& simulationData);
__device__ auto calculateCellIndex(glm::vec4 position,
                                   const Simulation::Parameters& simulationData,
                                   const SphSimulation::Grid& grid) -> glm::uvec3;
__device__ auto flattenCellIndex(glm::uvec3 cellIndex, glm::uvec3 gridSize) -> uint32_t;

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
    if (idx < grid.cellStartIndices.size)
    {
        grid.cellStartIndices.data[idx] = -1;
        grid.cellEndIndices.data[idx] = -1;
    }
}

__global__ void assignParticlesToCells(ParticlesData particles,
                                       SphSimulation::State state,
                                       Simulation::Parameters simulationData)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < particles.particleCount)
    {
        state.grid.particleArrayIndices.data[idx] = idx;
        const auto cellPosition = calculateCellIndex(particles.positions[idx], simulationData, state.grid);
        const auto cellIndex = flattenCellIndex(cellPosition, state.grid.gridSize);
        state.grid.particleGridIndices.data[idx] = cellIndex;
    }
}

__global__ void calculateCellStartAndEndIndices(SphSimulation::Grid grid)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= grid.particleArrayIndices.size)
    {
        return;
    }

    const auto cellIdx = grid.particleGridIndices.data[idx];

    if (idx == 0 || grid.particleGridIndices.data[idx - 1] != cellIdx)
    {
        grid.cellStartIndices.data[cellIdx] = idx;
    }
    if (idx == grid.particleArrayIndices.size - 1 || grid.particleGridIndices.data[idx + 1] != cellIdx)
    {
        grid.cellEndIndices.data[cellIdx] = idx;
    }
}

__global__ void computeDensities(ParticlesData particles,
                                 SphSimulation::State state,
                                 Simulation::Parameters simulationData)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }

    const auto position = particles.predictedPositions[idx];
    const auto originCell = calculateCellIndex(position, simulationData, state.grid);
    const auto radiusSquared = simulationData.smoothingRadius * simulationData.smoothingRadius;

    auto density = 0.F;
    auto nearDensity = 0.F;

    for (const auto offset : offsets)
    {
        const auto neighborCell = originCell + glm::uvec3 {offset.x, offset.y, offset.z};
        if (neighborCell.x >= state.grid.gridSize.x || neighborCell.y >= state.grid.gridSize.y ||
            neighborCell.z >= state.grid.gridSize.z)
        {
            continue;
        }
        const auto neighbourCellId = flattenCellIndex(neighborCell, state.grid.gridSize);
        const auto startIdx = state.grid.cellStartIndices.data[neighbourCellId];
        const auto endIdx = state.grid.cellEndIndices.data[neighbourCellId];

        if (startIdx == -1 || startIdx > endIdx)
        {
            continue;
        }

        for (auto i = startIdx; i <= endIdx; i++)
        {
            const auto neighbourIdx = state.grid.particleArrayIndices.data[i];
            const auto neighbourPosition = particles.predictedPositions[neighbourIdx];
            const auto offsetToNeighbour = neighbourPosition - position;
            const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);

            if (distanceSquared > radiusSquared)
            {
                continue;
            }
            const auto distance = glm::sqrt(distanceSquared);
            const auto neighbourMass = particles.masses[neighbourIdx];

            density += neighbourMass * device::densityKernel(distance, simulationData.smoothingRadius);
            nearDensity += neighbourMass * device::nearDensityKernel(distance, simulationData.smoothingRadius);
        }
    }
    particles.densities[idx] = density;
    particles.nearDensities[idx] = nearDensity;
}

__global__ void computePressureForce(ParticlesData particles,
                                     SphSimulation::State state,
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

    const auto originCell = calculateCellIndex(position, simulationData, state.grid);
    const auto radiusSquared = simulationData.smoothingRadius * simulationData.smoothingRadius;

    for (const auto offset : offsets)
    {
        const auto neighborCell = originCell + glm::uvec3 {offset.x, offset.y, offset.z};
        if (neighborCell.x >= state.grid.gridSize.x || neighborCell.y >= state.grid.gridSize.y ||
            neighborCell.z >= state.grid.gridSize.z)
        {
            continue;
        }
        const auto neighbourCellId = flattenCellIndex(neighborCell, state.grid.gridSize);
        const auto startIdx = state.grid.cellStartIndices.data[neighbourCellId];
        const auto endIdx = state.grid.cellEndIndices.data[neighbourCellId];

        if (startIdx == -1)
        {
            continue;
        }

        for (auto i = startIdx; i <= endIdx; i++)
        {
            const auto neighbourIdx = state.grid.particleArrayIndices.data[i];
            if (neighbourIdx == idx)
            {
                continue;
            }

            const auto neighbourPosition = particles.predictedPositions[neighbourIdx];
            const auto offsetToNeighbour = neighbourPosition - position;
            const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);

            if (distanceSquared > radiusSquared)
            {
                continue;
            }
            const auto densityNeighbour = particles.densities[neighbourIdx];
            const auto nearDensityNeighbour = particles.nearDensities[neighbourIdx];
            const auto pressureNeighbour =
                (densityNeighbour - simulationData.restDensity) * simulationData.pressureConstant;
            const auto nearPressureNeighbour = nearDensityNeighbour * simulationData.nearPressureConstant;

            const auto sharedPressure = (pressure + pressureNeighbour) / 2.F;
            const auto sharedNearPressure = (nearPressure + nearPressureNeighbour) / 2.F;

            const auto distance = glm::sqrt(distanceSquared);
            const auto direction = distance > 0.F ? offsetToNeighbour / distance : glm::vec4(0.F, 1.F, 0.F, 0.F);

            const auto neighbourMass = particles.masses[neighbourIdx];

            pressureForce += neighbourMass * direction *
                             device::densityDerivativeKernel(distance, simulationData.smoothingRadius) *
                             sharedPressure / densityNeighbour;
            pressureForce += neighbourMass * direction *
                             device::nearDensityDerivativeKernel(distance, simulationData.smoothingRadius) *
                             sharedNearPressure / nearPressureNeighbour;
        }
    }

    const auto particleMass = particles.masses[idx];
    const auto acceleration = pressureForce / (density * particleMass);

    particles.velocities[idx] += acceleration * dt;
}

__global__ void computeViscosityForce(ParticlesData particles,
                                      SphSimulation::State state,
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

    const auto originCell = calculateCellIndex(position, simulationData, state.grid);
    const auto radiusSquared = simulationData.smoothingRadius * simulationData.smoothingRadius;

    for (const auto offset : offsets)
    {
        const auto neighborCell = originCell + glm::uvec3 {offset.x, offset.y, offset.z};
        if (neighborCell.x >= state.grid.gridSize.x || neighborCell.y >= state.grid.gridSize.y ||
            neighborCell.z >= state.grid.gridSize.z)
        {
            continue;
        }
        const auto neighbourCellId = flattenCellIndex(neighborCell, state.grid.gridSize);
        const auto startIdx = state.grid.cellStartIndices.data[neighbourCellId];
        const auto endIdx = state.grid.cellEndIndices.data[neighbourCellId];

        if (startIdx == -1)
        {
            continue;
        }

        for (auto i = startIdx; i <= endIdx; i++)
        {
            const auto neighbourIdx = state.grid.particleArrayIndices.data[i];
            if (neighbourIdx == idx)
            {
                continue;
            }

            const auto neighbourPosition = particles.predictedPositions[neighbourIdx];
            const auto offsetToNeighbour = neighbourPosition - position;
            const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);

            if (distanceSquared > radiusSquared)
            {
                continue;
            }

            const auto distance = glm::sqrt(distanceSquared);
            const auto neighbourVelocity = particles.velocities[neighbourIdx];
            const auto neighbourMass = particles.masses[neighbourIdx];

            viscosityForce += neighbourMass * (neighbourVelocity - velocity) *
                              device::smoothingKernelPoly6(distance, simulationData.smoothingRadius);
        }
    }

    const auto particleMass = particles.masses[idx];
    particles.velocities[idx] += viscosityForce * simulationData.viscosityConstant * dt / particleMass;
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
        const auto minBoundary = simulationData.domain.min[i] + simulationData.particleRadius;
        const auto maxBoundary = simulationData.domain.max[i] - simulationData.particleRadius;

        if (particles.positions[id][i] < minBoundary)
        {
            particles.positions[id][i] = minBoundary;
            particles.velocities[id][i] = -particles.velocities[id][i] * simulationData.restitution;
        }

        if (particles.positions[id][i] > maxBoundary)
        {
            particles.positions[id][i] = maxBoundary;
            particles.velocities[id][i] = -particles.velocities[id][i] * simulationData.restitution;
        }
    }
}

__device__ auto calculateCellIndex(glm::vec4 position,
                                   const Simulation::Parameters& simulationData,
                                   const SphSimulation::Grid& grid) -> glm::uvec3
{
    const auto relativePosition = glm::vec3 {position} - simulationData.domain.min;
    const auto clampedPosition =
        glm::clamp(relativePosition, glm::vec3(0.F), simulationData.domain.max - simulationData.domain.min);

    return glm::uvec3 {clampedPosition.x / grid.cellSize.x,
                       clampedPosition.y / grid.cellSize.y,
                       clampedPosition.z / grid.cellSize.z};
}

__device__ auto flattenCellIndex(glm::uvec3 cellIndex, glm::uvec3 gridSize) -> uint32_t
{
    return cellIndex.x + (cellIndex.y * gridSize.x) + (cellIndex.z * gridSize.x * gridSize.y);
}

__global__ void computeExternalForces(ParticlesData particles, Simulation::Parameters simulationData, float deltaTime)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }

    particles.velocities[idx] += glm::vec4 {simulationData.gravity, 0.F} * deltaTime;
    particles.predictedPositions[idx] = particles.positions[idx] + particles.velocities[idx] * 1.F / 120.F;
}
}
