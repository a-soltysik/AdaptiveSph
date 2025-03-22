#pragma once

#include "cuda/Simulation.cuh"

namespace sph::cuda
{
__global__ void updatePositions(ParticleData* particles, int numObjects, float dt);
}
