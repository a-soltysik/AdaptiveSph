#include <device_atomic_functions.h>

#include <cstdint>
#include <cuda/simulation/Simulation.cuh>
#include <glm/exponential.hpp>
#include <glm/ext/vector_float4.hpp>

#include "Algorithm.cuh"
#include "Common.cuh"
#include "WendlandKernel.cuh"
#include "glm/geometric.hpp"
#include "simulation/SphSimulation.cuh"

namespace sph::cuda::kernel
{
namespace
{
__device__ auto getFluidNeighborSearchRadius(float smoothingRadius, NeighborGrid::Device grid) -> float
{
    return device::constant::wendlandRangeRatio * glm::max(smoothingRadius, grid.maxFluidSmoothingRadius);
}
}

__global__ void computeDensities(FluidParticlesData particles, NeighborGrid::Device grid)
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
                              getFluidNeighborSearchRadius(smoothingRadius, grid),
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
        getFluidNeighborSearchRadius(smoothingRadius, grid),
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

            if (distanceSquared > radiusSquared && distanceSquared < constants::minimalDistance)
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

    grid.forEachFluidNeighbor(
        position,
        particles.positions,
        getFluidNeighborSearchRadius(smoothingRadius, grid),
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

            if (distanceSquared > radiusSquared && distanceSquared < constants::minimalDistance)
            {
                return;
            }

            const auto neighborVelocity = particles.velocities[neighborIdx];
            const auto velocityDifference = velocity - neighborVelocity;
            const auto distance = glm::sqrt(distanceSquared);
            const auto neighborMass = particles.masses[neighborIdx];
            const auto neighborDensity = particles.densities[neighborIdx];

            const auto meanKernel = (device::wendlandDerivativeKernel(distance, neighborSmoothingRadius) +
                                     device::wendlandDerivativeKernel(distance, smoothingRadius)) /
                                    2.F;
            const auto meanDensity = (density + neighborDensity) / 2.F;
            const auto nominator = 4 * simulationData.viscosityConstant * distance * meanKernel;
            const auto denominator = meanDensity * distanceSquared;
            acceleration += velocityDifference * neighborMass * nominator / denominator;
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
        getFluidNeighborSearchRadius(smoothingRadius, grid),
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

            if (distanceSquared > radiusSquared && distanceSquared < constants::minimalDistance)
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

__global__ void computeExternalAccelerations(FluidParticlesData particles,
                                             Simulation::Parameters simulationData,
                                             ScenarioParameters scenarioParams,
                                             bool scenarioActivated)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }

    auto acceleration = glm::vec4 {simulationData.gravity, 0.F};

    if (scenarioActivated && scenarioParams.type == ScenarioType::PoiseuilleFlow &&
        scenarioParams.poiseuilleFlow.has_value())
    {
        acceleration.x += scenarioParams.poiseuilleFlow.value().drivingForce;
    }

    particles.accelerations[idx] = acceleration;
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
        getFluidNeighborSearchRadius(smoothingRadius, grid),
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

__global__ void updatePositions(FluidParticlesData particles, Simulation::Parameters::Domain domain, float dt)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }
    particles.positions[idx] += particles.velocities[idx] * dt;

    const auto position = glm::vec3 {particles.positions[idx]};
    const auto clampedPosition = glm::clamp(position, domain.min, domain.max);
    particles.positions[idx] = glm::vec4 {clampedPosition, particles.positions[idx].w};
}

__global__ void computeDensitiesPeriodic(FluidParticlesData particles, NeighborGrid::Device grid)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }

    const auto position = particles.positions[idx];
    const auto smoothingRadius = particles.smoothingRadii[idx];
    auto density = float {};

    grid.forEachFluidNeighborPeriodic(
        position,
        particles.positions,
        device::constant::wendlandRangeRatio * smoothingRadius,
        [position, &particles, &density](const auto neighbourIdx, const auto neighborPosition) {
            const auto offsetToNeighbour = neighborPosition - position;
            const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);

            const auto neighbourSmoothingRadius = particles.smoothingRadii[neighbourIdx];
            const auto radiusSquared = (device::constant::wendlandRangeRatio * neighbourSmoothingRadius) *
                                       (device::constant::wendlandRangeRatio * neighbourSmoothingRadius);

            if (distanceSquared > radiusSquared && distanceSquared < constants::minimalDistance)
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

__global__ void computePressureAccelerationsPeriodic(FluidParticlesData particles,
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

    grid.forEachFluidNeighborPeriodic(
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

            if (distanceSquared > radiusSquared && distanceSquared < constants::minimalDistance)
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

__global__ void computeViscosityAccelerationsPeriodic(FluidParticlesData particles,
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

    grid.forEachFluidNeighborPeriodic(
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

            if (distanceSquared > radiusSquared && distanceSquared < constants::minimalDistance)
            {
                return;
            }

            const auto neighborVelocity = particles.velocities[neighborIdx];
            const auto velocityDifference = velocity - neighborVelocity;
            const auto distance = glm::sqrt(distanceSquared);
            const auto neighborMass = particles.masses[neighborIdx];
            const auto neighborDensity = particles.densities[neighborIdx];

            const auto meanKernel = (device::wendlandDerivativeKernel(distance, neighborSmoothingRadius) +
                                     device::wendlandDerivativeKernel(distance, smoothingRadius)) /
                                    2.F;
            const auto meanDensity = (density + neighborDensity) / 2.F;
            const auto nominator = 4 * simulationData.viscosityConstant * distance * meanKernel;
            const auto denominator = meanDensity * distanceSquared;
            acceleration += velocityDifference * neighborMass * nominator / denominator;
        });

    particles.accelerations[idx] += acceleration;
}

__global__ void computeSurfaceTensionAccelerationsPeriodic(FluidParticlesData particles,
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

    grid.forEachFluidNeighborPeriodic(
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

            if (distanceSquared > radiusSquared && distanceSquared < constants::minimalDistance)
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

__global__ void updatePositionsPeriodic(FluidParticlesData particles, Simulation::Parameters::Domain domain, float dt)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }
    particles.positions[idx] += particles.velocities[idx] * dt;

    auto position = glm::vec3 {particles.positions[idx]};

    const auto domainSizeX = domain.max.x - domain.min.x;
    if (position.x < domain.min.x)
    {
        position.x += domainSizeX;
    }
    else if (position.x > domain.max.x)
    {
        position.x -= domainSizeX;
    }

    position.y = glm::clamp(position.y, domain.min.y, domain.max.y);
    position.z = glm::clamp(position.z, domain.min.z, domain.max.z);

    particles.positions[idx] = glm::vec4 {position, particles.positions[idx].w};
}

}
