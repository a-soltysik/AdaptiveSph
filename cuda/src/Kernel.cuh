#pragma once

#include <cstdint>

#include "cuda/Simulation.cuh"


namespace sph::cuda
{

__global__ void updatePositions(ParticleData* particles, uint32_t numObjects, float dt);
__global__ void handleCollisions(ParticleData* particles,
                                 uint32_t numObjects,
                                 Simulation::SimulationData simulationData);

__device__ void handleCollision(ParticleData& particle, const Simulation::SimulationData& simulationData);

}
