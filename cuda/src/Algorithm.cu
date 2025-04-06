#include <cuda/std/__iterator/distance.h>

#include <cstdint>
#include <glm/common.hpp>
#include <glm/ext/vector_uint3.hpp>
#include <thrust/system/detail/generic/distance.inl>

#include "Algorithm.cuh"
#include "Span.cuh"
#include "SphSimulation.cuh"
#include "cuda/Simulation.cuh"
#include "device/Kernel.cuh"
#include "glm/ext/vector_float3.hpp"
#include "glm/geometric.hpp"

namespace sph::cuda::kernel
{

__device__ void handleCollision(ParticleData& particle, const Simulation::Parameters& simulationData);
__device__ auto calculateCellIndex(glm::vec3 position,
                                   const Simulation::Parameters& simulationData,
                                   const SphSimulation::Grid& grid) -> glm::uvec3;
__device__ auto flattenCellIndex(glm::uvec3 cellIndex, glm::uvec3 gridSize) -> uint32_t;

__device__ void computeSurfaceTension(glm::vec3& surfaceTensionForce,
                                      ParticleData& particle,
                                      Span<ParticleData> particles,
                                      const SphSimulation::State& state,
                                      const Simulation::Parameters& simulationData,
                                      const glm::uvec3& cellPosition);

__global__ void handleCollisions(Span<ParticleData> particles, Simulation::Parameters simulationData)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < particles.size)
    {
        handleCollision(particles.data[idx], simulationData);
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

__global__ void assignParticlesToCells(Span<ParticleData> particles,
                                       SphSimulation::State state,
                                       Simulation::Parameters simulationData)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < particles.size)
    {
        state.grid.particleArrayIndices.data[idx] = idx;
        const auto cellPosition = calculateCellIndex(particles.data[idx].position, simulationData, state.grid);
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

__global__ void computeDensities(Span<ParticleData> particles,
                                 SphSimulation::State state,
                                 Simulation::Parameters simulationData)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.size)
    {
        return;
    }

    const auto position = particles.data[idx].predictedPosition;
    const auto originCell = calculateCellIndex(position, simulationData, state.grid);
    const auto radiusSquared = simulationData.smoothingRadius * simulationData.smoothingRadius;

    auto density = 0.F;
    auto nearDensity = 0.F;

    for (int dx = -1; dx <= 1; dx++)
    {
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dz = -1; dz <= 1; dz++)
            {
                const auto neighborCell = originCell + glm::uvec3(dx, dy, dz);
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
                    const auto neighbourPosition = particles.data[neighbourIdx].predictedPosition;
                    const auto offsetToNeighbour = neighbourPosition - position;
                    const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);

                    if (distanceSquared > radiusSquared)
                    {
                        continue;
                    }
                    const auto distance = glm::sqrt(distanceSquared);
                    density += device::densityKernel(distance, simulationData.smoothingRadius);
                    nearDensity += device::nearDensityKernel(distance, simulationData.smoothingRadius);
                }
            }
        }
    }
    particles.data[idx].density = density;
    particles.data[idx].nearDensity = nearDensity;
}

__global__ void computePressureForce(Span<ParticleData> particles,
                                     SphSimulation::State state,
                                     Simulation::Parameters simulationData,
                                     float dt)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.size)
    {
        return;
    }
    const auto position = particles.data[idx].predictedPosition;
    const auto density = particles.data[idx].density;
    const auto nearDensity = particles.data[idx].nearDensity;
    const auto pressure = (density - simulationData.restDensity) * simulationData.pressureConstant;
    const auto nearPressure = nearDensity * simulationData.nearPressureConstant;

    auto pressureForce = glm::vec3 {};

    const auto originCell = calculateCellIndex(position, simulationData, state.grid);
    const auto radiusSquared = simulationData.smoothingRadius * simulationData.smoothingRadius;

    for (int dx = -1; dx <= 1; dx++)
    {
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dz = -1; dz <= 1; dz++)
            {
                const auto neighborCell = originCell + glm::uvec3(dx, dy, dz);
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

                    const auto neighbourPosition = particles.data[neighbourIdx].predictedPosition;
                    const auto offsetToNeighbour = neighbourPosition - position;
                    const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);

                    if (distanceSquared > radiusSquared)
                    {
                        continue;
                    }
                    const auto densityNeighbour = particles.data[neighbourIdx].density;
                    const auto nearDensityNeighbour = particles.data[neighbourIdx].nearDensity;
                    const auto pressureNeighbour =
                        (densityNeighbour - simulationData.restDensity) * simulationData.pressureConstant;
                    const auto nearPressureNeighbour = nearDensityNeighbour * simulationData.nearPressureConstant;

                    const auto sharedPressure = (pressure + pressureNeighbour) / 2.F;
                    const auto sharedNearPressure = (nearPressure + nearPressureNeighbour) / 2.F;

                    const auto distance = glm::sqrt(distanceSquared);
                    const auto direction = distance > 0.F ? offsetToNeighbour / distance : glm::vec3(0.F, 1.F, 0.F);

                    pressureForce += direction *
                                     device::densityDerivativeKernel(distance, simulationData.smoothingRadius) *
                                     sharedPressure / densityNeighbour;
                    pressureForce += direction *
                                     device::nearDensityDerivativeKernel(distance, simulationData.smoothingRadius) *
                                     sharedNearPressure / nearPressureNeighbour;
                }
            }
        }
    }

    const auto acceleration = pressureForce / density;
    particles.data[idx].velocity += acceleration * dt;
}

__global__ void computeViscosityForce(Span<ParticleData> particles,
                                      SphSimulation::State state,
                                      Simulation::Parameters simulationData,
                                      float dt)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.size)
    {
        return;
    }
    const auto position = particles.data[idx].predictedPosition;
    const auto velocity = particles.data[idx].velocity;

    auto viscosityForce = glm::vec3 {};

    const auto originCell = calculateCellIndex(position, simulationData, state.grid);
    const auto radiusSquared = simulationData.smoothingRadius * simulationData.smoothingRadius;

    for (int dx = -1; dx <= 1; dx++)
    {
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dz = -1; dz <= 1; dz++)
            {
                const auto neighborCell = originCell + glm::uvec3(dx, dy, dz);
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

                    const auto neighbourPosition = particles.data[neighbourIdx].predictedPosition;
                    const auto offsetToNeighbour = neighbourPosition - position;
                    const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);

                    if (distanceSquared > radiusSquared)
                    {
                        continue;
                    }

                    const auto distance = glm::sqrt(distanceSquared);
                    const auto neighbourVelocity = particles.data[neighbourIdx].velocity;
                    viscosityForce += (neighbourVelocity - velocity) *
                                      device::smoothingKernelPoly6(distance, simulationData.smoothingRadius);
                }
            }
        }
    }

    particles.data[idx].velocity += viscosityForce * simulationData.viscosityConstant * dt;
}

__global__ void integrateMotion(Span<ParticleData> particles, Simulation::Parameters simulationData, float dt)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.size)
    {
        return;
    }

    const auto velocityMagnitude = glm::length(particles.data[idx].velocity);
    if (velocityMagnitude > simulationData.maxVelocity)
    {
        particles.data[idx].velocity *= simulationData.maxVelocity / velocityMagnitude;
    }
    particles.data[idx].position += particles.data[idx].velocity * dt;
}

__device__ void handleCollision(ParticleData& particle, const Simulation::Parameters& simulationData)
{
    for (int i = 0; i < 3; i++)
    {
        const auto minBoundary = simulationData.domain.min[i] + simulationData.particleRadius;
        const auto maxBoundary = simulationData.domain.max[i] - simulationData.particleRadius;

        if (particle.position[i] < minBoundary)
        {
            particle.position[i] = minBoundary;
            particle.velocity[i] = -particle.velocity[i] * simulationData.restitution;
        }

        if (particle.position[i] > maxBoundary)
        {
            particle.position[i] = maxBoundary;
            particle.velocity[i] = -particle.velocity[i] * simulationData.restitution;
        }
    }
}

__device__ auto calculateCellIndex(glm::vec3 position,
                                   const Simulation::Parameters& simulationData,
                                   const SphSimulation::Grid& grid) -> glm::uvec3
{
    const auto relativePosition = position - simulationData.domain.min;
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

__global__ void computeExternalForces(Span<ParticleData> particles,
                                      Simulation::Parameters simulationData,
                                      float deltaTime)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.size)
    {
        return;
    }

    particles.data[idx].velocity += simulationData.gravity * deltaTime;
    particles.data[idx].predictedPosition = particles.data[idx].position + particles.data[idx].velocity * 1.F / 120.F;
}

}
