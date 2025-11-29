#include <cstdint>
#include <cuda/simulation/Simulation.cuh>
#include <glm/exponential.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/geometric.hpp>

#include "BoundaryAlgorithm.cuh"
#include "Common.cuh"
#include "WendlandKernel.cuh"
#include "simulation/SphSimulation.cuh"

namespace sph::cuda::kernel
{

__global__ void computeBoundaryDensityContribution(FluidParticlesData fluidParticles,
                                                   BoundaryParticlesData boundaryParticles,
                                                   NeighborGrid::Device grid)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= fluidParticles.particleCount)
    {
        return;
    }

    const auto position = fluidParticles.positions[idx];
    const auto smoothingRadius = fluidParticles.smoothingRadii[idx];
    const auto radiusSquared = (device::constant::wendlandRangeRatio * smoothingRadius) *
                               (device::constant::wendlandRangeRatio * smoothingRadius);

    auto density = 0.0F;
    grid.forEachBoundaryNeighbor(
        position,
        boundaryParticles.positions,
        device::constant::wendlandRangeRatio * smoothingRadius,
        [position, smoothingRadius, radiusSquared, &boundaryParticles, &density](const auto boundaryIdx,
                                                                                 const auto boundaryPosition) {
            const auto offsetToNeighbour = boundaryPosition - position;
            const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);

            if (distanceSquared > radiusSquared)
            {
                return;
            }

            const auto distance = glm::sqrt(distanceSquared);
            const auto psi = boundaryParticles.psiValues[boundaryIdx];
            density += psi * device::wendlandKernel(distance, smoothingRadius);
        });

    fluidParticles.densities[idx] += density;
}

__global__ void computeBoundaryPressureAcceleration(FluidParticlesData fluidParticles,
                                                    BoundaryParticlesData boundaryParticles,
                                                    NeighborGrid::Device grid,
                                                    Simulation::Parameters simulationData)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= fluidParticles.particleCount)
    {
        return;
    }

    const auto position = fluidParticles.positions[idx];
    const auto density = fluidParticles.densities[idx];
    const auto smoothingRadius = fluidParticles.smoothingRadii[idx];

    const auto pressure = computeTaitPressure(density, simulationData.restDensity, simulationData.speedOfSound);
    const auto radiusSquared = (device::constant::wendlandRangeRatio * smoothingRadius) *
                               (device::constant::wendlandRangeRatio * smoothingRadius);
    const auto mass = fluidParticles.masses[idx];

    auto acceleration = glm::vec4();

    grid.forEachBoundaryNeighbor(
        position,
        boundaryParticles.positions,
        device::constant::wendlandRangeRatio * smoothingRadius,
        [position, density, pressure, smoothingRadius, radiusSquared, &boundaryParticles, &acceleration, mass](
            const auto boundaryIdx,
            const auto boundaryPosition) {
            const auto offsetToNeighbour = position - boundaryPosition;
            const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);

            if (distanceSquared > radiusSquared)
            {
                return;
            }

            const auto distance = glm::sqrt(distanceSquared);
            const auto direction = distance > 0.F ? offsetToNeighbour / distance : glm::vec4(0.F, 1.F, 0.F, 0.F);
            const auto psi = boundaryParticles.psiValues[boundaryIdx];
            const auto pressureTerm = pressure / (density * density);

            acceleration -=
                psi * pressureTerm * device::wendlandDerivativeKernel(distance, smoothingRadius) * direction;
        });

    fluidParticles.accelerations[idx] += acceleration;
}

__global__ void computeBoundaryFrictionAcceleration(FluidParticlesData fluidParticles,
                                                    BoundaryParticlesData boundaryParticles,
                                                    NeighborGrid::Device grid,
                                                    Simulation::Parameters simulationData)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= fluidParticles.particleCount)
    {
        return;
    }

    const auto position = fluidParticles.positions[idx];
    const auto velocity = fluidParticles.velocities[idx];
    const auto density = fluidParticles.densities[idx];
    const auto smoothingRadius = fluidParticles.smoothingRadii[idx];
    const auto radiusSquared = (device::constant::wendlandRangeRatio * smoothingRadius) *
                               (device::constant::wendlandRangeRatio * smoothingRadius);

    auto acceleration = glm::vec4 {};

    grid.forEachBoundaryNeighbor(
        position,
        boundaryParticles.positions,
        device::constant::wendlandRangeRatio * smoothingRadius,
        [position,
         velocity,
         density,
         smoothingRadius,
         radiusSquared,
         &simulationData,
         &boundaryParticles,
         &acceleration](const auto boundaryIdx, const auto boundaryPosition) {
            const auto offsetToNeighbour = position - boundaryPosition;
            const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);

            if (distanceSquared > radiusSquared)
            {
                return;
            }

            const auto distance = glm::sqrt(distanceSquared);

            const auto psi = boundaryParticles.psiValues[boundaryIdx];
            const auto frictionCoefficient = boundaryParticles.viscosityCoefficients[boundaryIdx];
            const auto velocityDifference = velocity;
            const auto compression = glm::dot(velocityDifference, offsetToNeighbour);
            if (compression >= 0.F)
            {
                return;
            }

            static constexpr auto epsilon = 0.01F;
            const auto nu = (frictionCoefficient * smoothingRadius * simulationData.speedOfSound) / (2.0F * density);
            const auto pi = -nu * compression / (distanceSquared + epsilon * smoothingRadius * smoothingRadius);
            const auto direction = distance > 0.F ? offsetToNeighbour / distance : glm::vec4(0.F, 1.F, 0.F, 0.F);

            acceleration -= direction * psi * pi * device::wendlandDerivativeKernel(distance, smoothingRadius);
        });

    fluidParticles.accelerations[idx] += acceleration;
}
}
