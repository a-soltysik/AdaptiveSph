#pragma once

#include "SphSimulation.cuh"
#include "cuda/Simulation.cuh"

namespace sph::cuda::kernel
{

__global__ void handleCollisions(Span<ParticleData> particles, Simulation::Parameters simulationData);
__global__ void resetGrid(SphSimulation::Grid grid);
__global__ void assignParticlesToCells(Span<ParticleData> particles,
                                       SphSimulation::State state,
                                       Simulation::Parameters simulationData);
__global__ void calculateCellStartAndEndIndices(SphSimulation::Grid grid);
__global__ void computeDensities(Span<ParticleData> particles,
                                 SphSimulation::State state,
                                 Simulation::Parameters simulationData);
__global__ void computePressureForce(Span<ParticleData> particles,
                                     SphSimulation::State state,
                                     Simulation::Parameters simulationData,
                                     float dt);
__global__ void computeViscosityForce(Span<ParticleData> particles,
                                      SphSimulation::State state,
                                      Simulation::Parameters simulationData,
                                      float dt);
__global__ void integrateMotion(Span<ParticleData> particles, Simulation::Parameters simulationData, float dt);
__global__ void computeExternalForces(Span<ParticleData> particles,
                                      Simulation::Parameters simulationData,
                                      float deltaTime);
}
