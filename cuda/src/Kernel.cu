#include "Kernel.cuh"

namespace sph::cuda
{
__global__ void updatePositions(ParticleData* particles, int numObjects, float dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numObjects)
    {
        particles[idx].velocity += particles[idx].force / particles[idx].mass * dt;
        particles[idx].position += particles[idx].velocity * dt;
    }
}
}
