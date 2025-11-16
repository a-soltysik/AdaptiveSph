#include <device_atomic_functions.h>

#include <cstdint>
#include <cuda/Simulation.cuh>
#include <glm/exponential.hpp>
#include <glm/ext/vector_float4.hpp>

#include "Algorithm.cuh"
#include "Common.cuh"
#include "glm/geometric.hpp"
#include "kernels/Kernel.cuh"
#include "simulation/SphSimulation.cuh"

namespace sph::cuda::kernel
{

__global__ void computeDensities(FluidParticlesData particles,
                                 NeighborGrid::Device grid,
                                 Simulation::Parameters simulationData)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }

    const auto position = particles.positions[idx];
    const auto smoothingRadius = particles.smoothingRadii[idx];
    auto density = float {};

    grid.forEachFluidNeighbor(position,
                              particles.positions,
                              device::constant::wendlandRangeRatio * smoothingRadius,
                              [position, &particles, &density](const auto neighbourIdx, const auto neighborPosition) {
                                  const auto offsetToNeighbour = neighborPosition - position;
                                  const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);

                                  const auto neighbourSmoothingRadius = particles.smoothingRadii[neighbourIdx];
                                  const auto radiusSquared =
                                      (device::constant::wendlandRangeRatio * neighbourSmoothingRadius) *
                                      (device::constant::wendlandRangeRatio * neighbourSmoothingRadius);

                                  if (distanceSquared > radiusSquared)
                                  {
                                      return;
                                  }

                                  const auto distance = glm::sqrt(distanceSquared);
                                  const auto neighbourMass = particles.masses[neighbourIdx];
                                  const auto kernel = device::wendlandKernel(distance, neighbourSmoothingRadius);

                                  density += neighbourMass * kernel;
                              });

    particles.densities[idx] = density;
}

__global__ void computePressureAccelerations(FluidParticlesData particles,
                                             NeighborGrid::Device grid,
                                             Simulation::Parameters simulationData)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }
    const auto position = particles.positions[idx];
    const auto density = particles.densities[idx];
    const auto smoothingRadius = particles.smoothingRadii[idx];
    const auto pressure = computeTaitPressure(density, simulationData.restDensity, simulationData.speedOfSound);

    auto acceleration = glm::vec4 {};

    grid.forEachFluidNeighbor(
        position,
        particles.positions,
        device::constant::wendlandRangeRatio * smoothingRadius,
        [position, &particles, &simulationData, pressure, density, &acceleration, idx](const auto neighbourIdx,
                                                                                       const auto neighborPosition) {
            if (neighbourIdx == idx)
            {
                return;
            }

            const auto offsetToNeighbour = position - neighborPosition;
            const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);
            const auto neighbourSmoothingRadius = particles.smoothingRadii[neighbourIdx];
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

__global__ void computeViscosityAccelerations(FluidParticlesData particles,
                                              NeighborGrid::Device grid,
                                              Simulation::Parameters simulationData)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }

    const auto position = particles.positions[idx];
    const auto velocity = particles.velocities[idx];
    const auto smoothingRadius = particles.smoothingRadii[idx];
    const auto density = particles.densities[idx];

    auto acceleration = glm::vec4 {};

    static constexpr auto epsilon = 0.01F;

    grid.forEachFluidNeighbor(
        position,
        particles.positions,
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
            const auto neighborSmoothingRadius = particles.smoothingRadii[neighborIdx];
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

__global__ void computeSurfaceTensionAccelerations(FluidParticlesData particles,
                                                   NeighborGrid::Device grid,
                                                   Simulation::Parameters simulationData)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }

    const auto position = particles.positions[idx];
    const auto smoothingRadius = particles.smoothingRadii[idx];
    const auto mass = particles.masses[idx];

    auto acceleration = glm::vec4 {};

    grid.forEachFluidNeighbor(
        position,
        particles.positions,
        device::constant::wendlandRangeRatio * smoothingRadius,
        [position, &particles, &acceleration, idx](const auto neighborIdx, const glm::vec4& neighborPosition) {
            if (neighborIdx == idx)
            {
                return;
            }

            const auto offsetToNeighbour = position - neighborPosition;
            const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);
            const auto neighborSmoothingRadius = particles.smoothingRadii[neighborIdx];
            const auto radiusSquared = (device::constant::wendlandRangeRatio * neighborSmoothingRadius) *
                                       (device::constant::wendlandRangeRatio * neighborSmoothingRadius);

            if (distanceSquared > radiusSquared)
            {
                return;
            }

            const auto distance = glm::sqrt(distanceSquared);
            const auto neighborMass = particles.masses[neighborIdx];
            const auto direction = distance > 0.F ? offsetToNeighbour / distance : glm::vec4(0.F, 1.F, 0.F, 0.F);

            acceleration += neighborMass * device::wendlandKernel(distance, neighborSmoothingRadius) * direction;
        });

    particles.accelerations[idx] += (simulationData.surfaceTensionConstant / mass) * acceleration;
}

__global__ void computeExternalAccelerations(FluidParticlesData particles, Simulation::Parameters simulationData)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }

    particles.accelerations[idx] = glm::vec4 {simulationData.gravity, 0.F};
}

__global__ void sumAllNeighbors(FluidParticlesData particles, NeighborGrid::Device grid, uint32_t* totalNeighborCount)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }
    const auto position = particles.positions[idx];
    const auto smoothingRadius = particles.smoothingRadii[idx];
    auto count = uint32_t {0};

    grid.forEachFluidNeighbor(
        position,
        particles.positions,
        device::constant::wendlandRangeRatio * smoothingRadius,
        [position, &particles, &count, idx](const auto neighborIdx, const glm::vec4& adjustedPos) {
            if (idx == neighborIdx)
            {
                return;
            }

            const auto offsetToNeighbour = adjustedPos - position;
            const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);
            const auto neighborSmoothingRadius = particles.smoothingRadii[neighborIdx];
            const auto radiusSquared = (device::constant::wendlandRangeRatio * neighborSmoothingRadius) *
                                       (device::constant::wendlandRangeRatio * neighborSmoothingRadius);

            if (distanceSquared <= radiusSquared)
            {
                count++;
            }
        });

    atomicAdd(totalNeighborCount, count);
}

__global__ void halfKickVelocities(FluidParticlesData particles, Simulation::Parameters simulationData, float halfDt)
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

__global__ void updatePositions(FluidParticlesData particles, float dt)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }
    particles.positions[idx] += particles.velocities[idx] * dt;
}

}
