#include <cmath>
#include <cstdint>
#include <cuda/Simulation.cuh>
#include <glm/exponential.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>

#include "Algorithm.cuh"
#include "glm/ext/scalar_constants.hpp"
#include "glm/geometric.hpp"
#include "kernels/Kernel.cuh"
#include "simulation/adaptive/SphSimulation.cuh"
#include "utils/Iteration.cuh"
#include "utils/Utils.cuh"

namespace sph::cuda::kernel
{

__device__ void handleCollision(ParticlesData particles, uint32_t id, const Simulation::Parameters& simulationData);
__device__ void handleLidDrivenCavityCollision(ParticlesData particles,
                                               uint32_t id,
                                               const Simulation::Parameters& simulationData);
__device__ void handlePoiseuilleFlowCollision(ParticlesData particles,
                                              uint32_t id,
                                              const Simulation::Parameters& simulationData);
__device__ void handleTaylorGreenVortexCollision(ParticlesData particles,
                                                 uint32_t id,
                                                 const Simulation::Parameters& simulationData);
__device__ void handleStandardCollision(ParticlesData particles,
                                        uint32_t id,
                                        const Simulation::Parameters& simulationData);
__device__ void handleNoSlipBoundaries(ParticlesData particles,
                                       uint32_t id,
                                       const Simulation::Parameters& simulationData,
                                       int axis);
__device__ void handleStandardBoundariesForAxis(ParticlesData particles,
                                                uint32_t id,
                                                const Simulation::Parameters& simulationData,
                                                int axis);
__device__ void handlePeriodicBoundariesForAxis(ParticlesData particles,
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
                                       SphSimulation::State state,
                                       Simulation::Parameters simulationData)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < particles.particleCount)
    {
        state.grid.particleArrayIndices[idx] = idx;
        const auto cellPosition = calculateCellIndex(particles.predictedPositions[idx], simulationData, state.grid);
        const auto cellIndex = flattenCellIndex(cellPosition, state.grid.gridSize);
        state.grid.particleGridIndices[idx] = cellIndex;
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
                                 SphSimulation::State state,
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
                     state.grid,
                     [&](const auto neighbourIdx, const glm::vec4& adjustedPos) {
                         //if (neighbourIdx == idx)
                         //{
                         //    return;
                         //}
                         const auto offsetToNeighbour = adjustedPos - position;
                         const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);

                         const auto neighbourSmoothingRadius = particles.smoothingRadiuses[neighbourIdx];
                         const auto radiusSquared = 4 * neighbourSmoothingRadius * neighbourSmoothingRadius;

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

    forEachNeighbour(
        position,
        particles,
        simulationData,
        state.grid,
        [&](const auto neighbourIdx, const glm::vec4& adjustedPos) {
            if (neighbourIdx == idx)
            {
                return;
            }

            const auto offsetToNeighbour = adjustedPos - position;
            const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);
            const auto neighbourSmoothingRadius = particles.smoothingRadiuses[neighbourIdx];
            const auto radiusSquared = 4 * neighbourSmoothingRadius * neighbourSmoothingRadius;

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
                             device::densityDerivativeKernel(distance, neighbourSmoothingRadius) * sharedPressure /
                             densityNeighbour;
            pressureForce += neighbourMass * direction *
                             device::nearDensityDerivativeKernel(distance, neighbourSmoothingRadius) *
                             sharedNearPressure / nearDensityNeighbour;
            //pressureForce +=
            //    direction * neighbourMass *
            //    (pressure / (density * density) + pressureNeighbour / (densityNeighbour * densityNeighbour)) *
            //    device::densityDerivativeKernel(distance, neighbourSmoothingRadius);
            //
            //pressureForce += direction * neighbourMass *
            //                 (nearPressure / (nearDensity * nearDensity) +
            //                  nearPressureNeighbour / (nearDensityNeighbour * nearDensityNeighbour)) *
            //                 device::nearDensityDerivativeKernel(distance, neighbourSmoothingRadius);
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

    forEachNeighbour(position,
                     particles,
                     simulationData,
                     state.grid,
                     [&](const auto neighbourIdx, const glm::vec4& adjustedPos) {
                         if (neighbourIdx == idx)
                         {
                             return;
                         }

                         const auto offsetToNeighbour = adjustedPos - position;
                         const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);
                         const auto smoothingRadius = particles.smoothingRadiuses[neighbourIdx];
                         const auto radiusSquared = smoothingRadius * smoothingRadius;

                         if (distanceSquared > radiusSquared)
                         {
                             return;
                         }

                         const auto distance = glm::sqrt(distanceSquared);
                         const auto neighbourVelocity = particles.velocities[neighbourIdx];
                         const auto neighbourMass = particles.masses[neighbourIdx];
                         const auto neighbourDensity = particles.densities[neighbourIdx];

                         viscosityForce += neighbourMass * (neighbourVelocity - velocity) / neighbourDensity *
                                           device::viscosityLaplacianKernel(distance, smoothingRadius);
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
    switch (simulationData.testCase)
    {
    case Simulation::Parameters::TestCase::LidDrivenCavity:
        handleLidDrivenCavityCollision(particles, id, simulationData);
        break;
    case Simulation::Parameters::TestCase::PoiseuilleFlow:
        handlePoiseuilleFlowCollision(particles, id, simulationData);
        break;
    case Simulation::Parameters::TestCase::TaylorGreenVortex:
        handleTaylorGreenVortexCollision(particles, id, simulationData);
        break;
    default:
        handleStandardCollision(particles, id, simulationData);
        break;
    }
}

__device__ void handleLidDrivenCavityCollision(ParticlesData particles,
                                               uint32_t id,
                                               const Simulation::Parameters& simulationData)
{
    for (int i = 0; i < 3; i++)
    {
        const auto minBoundary = simulationData.domain.min[i] + particles.radiuses[id];
        const auto maxBoundary = simulationData.domain.max[i] - particles.radiuses[id];

        if (particles.positions[id][i] < minBoundary)
        {
            particles.positions[id][i] = minBoundary;
            if (i == 1)
            {
                // Top wall moves at lid velocity
                particles.velocities[id] = glm::vec4(simulationData.lidVelocity, 0.5F, 0.0F, 0.0F);
            }
            else
            {
                particles.velocities[id][i] = -particles.velocities[id][i] * simulationData.restitution;
            }
        }

        if (particles.positions[id][i] > maxBoundary)
        {
            particles.positions[id][i] = maxBoundary;
            particles.velocities[id][i] = -particles.velocities[id][i] * simulationData.restitution;
        }
    }
}

__device__ void handlePoiseuilleFlowCollision(ParticlesData particles,
                                              uint32_t id,
                                              const Simulation::Parameters& simulationData)
{
    // Handle y-axis (channel height) - no-slip boundaries
    handleNoSlipBoundaries(particles, id, simulationData, 1);

    // Handle z-axis (channel width) - standard boundaries
    handleStandardBoundariesForAxis(particles, id, simulationData, 2);

    // Handle x-axis (flow direction) - periodic boundaries
    handlePeriodicBoundariesForAxis(particles, id, simulationData, 0);
}

__device__ void handleTaylorGreenVortexCollision(ParticlesData particles,
                                                 uint32_t id,
                                                 const Simulation::Parameters& simulationData)
{
    const auto domainMin = simulationData.domain.min;
    const auto domainMax = simulationData.domain.max;
    const auto domainSize = domainMax - domainMin;

    // Handle periodic boundaries in all directions
    for (int i = 0; i < 3; i++)
    {
        if (particles.positions[id][i] < domainMin[i])
        {
            particles.positions[id][i] += domainSize[i];
            particles.predictedPositions[id][i] += domainSize[i];
        }
        else if (particles.positions[id][i] >= domainMax[i])
        {
            particles.positions[id][i] -= domainSize[i];
            particles.predictedPositions[id][i] -= domainSize[i];
        }
    }
}

__device__ void handleStandardCollision(ParticlesData particles,
                                        uint32_t id,
                                        const Simulation::Parameters& simulationData)
{
    for (int i = 0; i < 3; i++)
    {
        handleStandardBoundariesForAxis(particles, id, simulationData, i);
    }
}

__device__ void handleNoSlipBoundaries(ParticlesData particles,
                                       uint32_t id,
                                       const Simulation::Parameters& simulationData,
                                       int axis)
{
    const auto minBoundary = simulationData.domain.min[axis] + particles.radiuses[id];
    const auto maxBoundary = simulationData.domain.max[axis] - particles.radiuses[id];

    if (particles.positions[id][axis] < minBoundary)
    {
        particles.positions[id][axis] = minBoundary;
        particles.velocities[id] = -particles.velocities[id] * 0.2F;
    }

    if (particles.positions[id][axis] > maxBoundary)
    {
        particles.positions[id][axis] = maxBoundary;
        particles.velocities[id] = -particles.velocities[id] * 0.2F;
    };
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

__device__ void handlePeriodicBoundariesForAxis(ParticlesData particles,
                                                uint32_t id,
                                                const Simulation::Parameters& simulationData,
                                                int axis)
{
    const auto minBoundary = simulationData.domain.min[axis] + particles.radiuses[id];
    const auto maxBoundary = simulationData.domain.max[axis] - particles.radiuses[id];
    const auto domainLength =
        simulationData.domain.max[axis] - simulationData.domain.min[axis] - (2.F * particles.radiuses[id]);

    if (particles.positions[id][axis] < minBoundary)
    {
        // Move particle to the other end of the domain
        particles.positions[id][axis] += domainLength;
    }

    if (particles.positions[id][axis] > maxBoundary)
    {
        // Move particle to the beginning of the domain
        particles.positions[id][axis] -= domainLength;
    }
}

__global__ void computeExternalForces(ParticlesData particles, Simulation::Parameters simulationData, float deltaTime)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }

    // Apply gravity force to velocity
    particles.velocities[idx] += glm::vec4 {simulationData.gravity, 0.F} * deltaTime;
    // Special handling for Taylor-Green vortex - initialize velocity field
    if (simulationData.testCase == cuda::Simulation::Parameters::TestCase::TaylorGreenVortex)
    {
        // Only set velocity at the first time step - check if velocity is zero
        if (glm::length(glm::vec3(particles.velocities[idx])) < 1e-6F)
        {
            const auto pos = particles.positions[idx];
            const auto domainMin = simulationData.domain.min;
            const auto domainSize = simulationData.domain.max - domainMin;
            // Map position to [0, 2pi] range for Taylor-Green equations
            const auto x = ((pos.x - domainMin.x) / domainSize.x) * 2.0F * glm::pi<float>();
            const auto y = ((pos.y - domainMin.y) / domainSize.y) * 2.0F * glm::pi<float>();
            const auto z = ((pos.z - domainMin.z) / domainSize.z) * 2.0F * glm::pi<float>();
            // Calculate Taylor-Green velocity field
            const auto u = std::cos(x) * std::sin(y) * std::cos(z);
            const auto v = -std::sin(x) * std::cos(y) * std::cos(z);
            const auto w = 0.0F;

            // Set the velocity
            particles.velocities[idx] = glm::vec4(u, v, w, 0.0F);
        }
    }

    // Update predicted positions
    particles.predictedPositions[idx] = particles.positions[idx] + particles.velocities[idx] * deltaTime;
}

__global__ void countNeighbors(ParticlesData particles,
                               SphSimulation::State state,
                               Simulation::Parameters simulationData,
                               uint32_t* neighborCounts)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }
    const auto position = particles.positions[idx];
    const auto radiusSquared = particles.smoothingRadiuses[idx] * particles.smoothingRadiuses[idx];
    uint32_t count = 0;

    forEachNeighbour(position,
                     particles,
                     simulationData,
                     state.grid,
                     [&](const auto neighbourIdx, const glm::vec4& adjustedPos) {
                         if (idx == neighbourIdx)
                         {
                             return;
                         }

                         // Use the adjustedPos for distance calculation
                         const auto offsetToNeighbour = adjustedPos - position;
                         const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);

                         if (distanceSquared <= radiusSquared)
                         {
                             count++;
                         }
                     });

    neighborCounts[idx] = count;
}

__global__ void calculateDensityDeviations(ParticlesData particles, float restDensity)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }

    const auto deviation = (particles.densities[idx] - restDensity) / restDensity;
    particles.densityDeviations[idx] = deviation;
}
}
