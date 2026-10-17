#pragma once

#include <cstdint>
#include <cuda/simulation/Simulation.cuh>

#include "simulation/SphSimulation.cuh"
#include "simulation/neighbors/NeighborGrid.cuh"

namespace sph::cuda::kernel
{

__global__ void computeDensities(FluidParticlesData particles, NeighborGrid::Device grid);
__global__ void computeDensitiesPeriodic(FluidParticlesData particles, NeighborGrid::Device grid);

__global__ void computePressureAccelerations(FluidParticlesData particles,
                                             NeighborGrid::Device grid,
                                             Simulation::Parameters simulationData);
__global__ void computePressureAccelerationsPeriodic(FluidParticlesData particles,
                                                     NeighborGrid::Device grid,
                                                     Simulation::Parameters simulationData);

__global__ void computeViscosityAccelerations(FluidParticlesData particles,
                                              NeighborGrid::Device grid,
                                              Simulation::Parameters simulationData);
__global__ void computeViscosityAccelerationsPeriodic(FluidParticlesData particles,
                                                      NeighborGrid::Device grid,
                                                      Simulation::Parameters simulationData);

__global__ void computeSurfaceTensionAccelerations(FluidParticlesData particles,
                                                   NeighborGrid::Device grid,
                                                   Simulation::Parameters simulationData);
__global__ void computeSurfaceTensionAccelerationsPeriodic(FluidParticlesData particles,
                                                           NeighborGrid::Device grid,
                                                           Simulation::Parameters simulationData);

__global__ void computeExternalAccelerations(FluidParticlesData particles,
                                             Simulation::Parameters simulationData,
                                             ScenarioParameters scenarioParams,
                                             bool scenarioActivated);

__global__ void halfKickVelocities(FluidParticlesData particles, Simulation::Parameters simulationData, float halfDt);
__global__ void updatePositions(FluidParticlesData particles, Simulation::Parameters::Domain domain, float dt);
__global__ void updatePositionsPeriodic(FluidParticlesData particles, Simulation::Parameters::Domain domain, float dt);

__global__ void sumAllNeighbors(FluidParticlesData particles, NeighborGrid::Device grid, uint32_t* totalNeighborCount);
}
