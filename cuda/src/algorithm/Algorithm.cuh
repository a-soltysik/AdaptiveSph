#pragma once

#include <cstdint>
#include <cuda/Simulation.cuh>

#include "simulation/adaptive/SphSimulation.cuh"

namespace sph::cuda::kernel
{

__global__ void handleCollisions(ParticlesData particles, Simulation::Parameters simulationData);
__global__ void resetGrid(SphSimulation::Grid grid);
__global__ void assignParticlesToCells(ParticlesData particles,
                                       SphSimulation::State state,
                                       Simulation::Parameters simulationData);
__global__ void calculateCellStartAndEndIndices(SphSimulation::Grid grid, uint32_t particleCount);
__global__ void computeDensities(ParticlesData particles,
                                 SphSimulation::State state,
                                 Simulation::Parameters simulationData);
__global__ void computePressureForce(ParticlesData particles,
                                     SphSimulation::State state,
                                     Simulation::Parameters simulationData,
                                     float dt);
__global__ void computeViscosityForce(ParticlesData particles,
                                      SphSimulation::State state,
                                      Simulation::Parameters simulationData,
                                      float dt);
__global__ void integrateMotion(ParticlesData particles, Simulation::Parameters simulationData, float dt);
__global__ void computeExternalForces(ParticlesData particles, Simulation::Parameters simulationData, float deltaTime);
__global__ void countNeighbors(ParticlesData particles,
                               SphSimulation::State state,
                               Simulation::Parameters simulationData,
                               uint32_t* neighborCounts);

__global__ void calculateDensityDeviations(ParticlesData particles, float restDensity);
}
