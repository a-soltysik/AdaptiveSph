#pragma once

#include <cstdint>
#include <cuda/Simulation.cuh>

#include "simulation/SphSimulation.cuh"

namespace sph::cuda::kernel
{

__global__ void resetGrid(SphSimulation::Grid grid);
__global__ void assignParticlesToCells(ParticlesData particles,
                                       SphSimulation::Grid grid,
                                       Simulation::Parameters simulationData);
__global__ void calculateCellStartAndEndIndices(SphSimulation::Grid grid, uint32_t particleCount);

__global__ void computeDensities(ParticlesData particles,
                                 SphSimulation::Grid grid,
                                 Simulation::Parameters simulationData);
__global__ void computePressureAccelerations(ParticlesData particles,
                                             SphSimulation::Grid grid,
                                             Simulation::Parameters simulationData);
__global__ void computeViscosityAccelerations(ParticlesData particles,
                                              SphSimulation::Grid grid,
                                              Simulation::Parameters simulationData);
__global__ void computeSurfaceTensionAccelerations(ParticlesData particles,
                                                   SphSimulation::Grid grid,
                                                   Simulation::Parameters simulationData);
__global__ void computeExternalAccelerations(ParticlesData particles, Simulation::Parameters simulationData);

__global__ void halfKickVelocities(ParticlesData particles, Simulation::Parameters simulationData, float halfDt);
__global__ void updatePositions(ParticlesData particles, float dt);
__global__ void handleCollisions(ParticlesData particles, Simulation::Parameters simulationData);

__global__ void sumAllNeighbors(ParticlesData particles,
                                SphSimulation::Grid grid,
                                Simulation::Parameters simulationData,
                                uint32_t* totalNeighborCount);
}
