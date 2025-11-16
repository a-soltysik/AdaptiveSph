#pragma once

#include <cstdint>
#include <cuda/Simulation.cuh>

#include "NeighborGrid.cuh"
#include "simulation/SphSimulation.cuh"

namespace sph::cuda::kernel
{

__global__ void computeDensities(FluidParticlesData particles,
                                 NeighborGrid::Device grid,
                                 Simulation::Parameters simulationData);
__global__ void computePressureAccelerations(FluidParticlesData particles,
                                             NeighborGrid::Device grid,
                                             Simulation::Parameters simulationData);
__global__ void computeViscosityAccelerations(FluidParticlesData particles,
                                              NeighborGrid::Device grid,
                                              Simulation::Parameters simulationData);
__global__ void computeSurfaceTensionAccelerations(FluidParticlesData particles,
                                                   NeighborGrid::Device grid,
                                                   Simulation::Parameters simulationData);
__global__ void computeExternalAccelerations(FluidParticlesData particles, Simulation::Parameters simulationData);

__global__ void halfKickVelocities(FluidParticlesData particles, Simulation::Parameters simulationData, float halfDt);
__global__ void updatePositions(FluidParticlesData particles, float dt);

__global__ void sumAllNeighbors(FluidParticlesData particles, NeighborGrid::Device grid, uint32_t* totalNeighborCount);
}
