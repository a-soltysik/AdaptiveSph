#include <cstdint>
#include <cuda/Simulation.cuh>
#include <glm/ext/vector_uint3.hpp>

#include "Algorithm.cuh"
#include "Span.cuh"
#include "SphSimulation.cuh"
#include "common/Iteration.hpp"
#include "common/Utils.cuh"
#include "device/Kernel.cuh"
#include "glm/geometric.hpp"

namespace sph::cuda::kernel
{

__device__ void handleCollision(ParticlesData particles, uint32_t id, const Simulation::Parameters& simulationData);

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

__global__ void calculateCellStartAndEndIndices(SphSimulation::Grid grid, uint32_t particleCount)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particleCount)
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
    const auto radiusSquared = particles.smoothingRadiuses[idx] * particles.smoothingRadiuses[idx];

    auto density = 0.F;
    auto nearDensity = 0.F;

    forEachNeighbour(position, simulationData, state.grid, [&](const auto neighbourIdx) {
        const auto neighbourPosition = particles.predictedPositions[neighbourIdx];
        const auto offsetToNeighbour = neighbourPosition - position;
        const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);

        if (distanceSquared > radiusSquared)
        {
            return;
        }
        const auto distance = glm::sqrt(distanceSquared);
        const auto neighbourMass = particles.masses[neighbourIdx];

        density += neighbourMass * device::densityKernel(distance, particles.smoothingRadiuses[idx]);
        nearDensity += neighbourMass * device::nearDensityKernel(distance, particles.smoothingRadiuses[idx]);
    });

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
    const auto radiusSquared = particles.smoothingRadiuses[idx] * particles.smoothingRadiuses[idx];

    forEachNeighbour(position, simulationData, state.grid, [&](const auto neighbourIdx) {
        if (neighbourIdx == idx)
        {
            return;
        }

        const auto neighbourPosition = particles.predictedPositions[neighbourIdx];
        const auto offsetToNeighbour = neighbourPosition - position;
        const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);

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
        const auto direction = distance > 0.F ? offsetToNeighbour / distance : glm::vec4(0.F, 1.F, 0.F, 0.F);

        const auto neighbourMass = particles.masses[neighbourIdx];

        pressureForce += neighbourMass * direction *
                         device::densityDerivativeKernel(distance, particles.smoothingRadiuses[idx]) * sharedPressure /
                         densityNeighbour;
        pressureForce += neighbourMass * direction *
                         device::nearDensityDerivativeKernel(distance, particles.smoothingRadiuses[idx]) *
                         sharedNearPressure / nearDensityNeighbour;
    });

    const auto particleMass = particles.masses[idx];
    const auto acceleration = pressureForce / particleMass;

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

    const auto radiusSquared = particles.smoothingRadiuses[idx] * particles.smoothingRadiuses[idx];

    forEachNeighbour(position, simulationData, state.grid, [&](const auto neighbourIdx) {
        if (neighbourIdx == idx)
        {
            return;
        }

        const auto neighbourPosition = particles.predictedPositions[neighbourIdx];
        const auto offsetToNeighbour = neighbourPosition - position;
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
    particles.velocities[idx] += acceleration * dt / particleMass;
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
        const auto minBoundary = simulationData.domain.min[i] + particles.radiuses[id];
        const auto maxBoundary = simulationData.domain.max[i] - particles.radiuses[id];

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
