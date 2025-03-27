#include <cstdint>

#include "Kernel.cuh"
#include "cuda/Simulation.cuh"
#include "glm/detail/func_geometric.inl"

namespace sph::cuda
{

__global__ void updatePositions(ParticleData* particles, uint32_t numObjects, float dt)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < numObjects)
    {
        particles[idx].velocity += particles[idx].force / particles[idx].mass * dt;
        particles[idx].position += particles[idx].velocity * dt;
    }
}

__global__ void handleCollisions(ParticleData* particles,
                                 uint32_t numObjects,
                                 Simulation::SimulationData simulationData)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < numObjects)
    {
        handleCollision(particles[idx], simulationData);
    }
}

__device__ void handleCollision(ParticleData& particle, const Simulation::SimulationData& simulationData)
{
    for (int i = 0; i < 3; i++)
    {
        const auto minBoundary = simulationData.domain.min[i] + particle.mass;
        const auto maxBoundary = simulationData.domain.max[i] - particle.mass;

        if (particle.position[i] < minBoundary)
        {
            particle.position[i] = minBoundary;
            particle.velocity[i] = -particle.velocity[i] * simulationData.restitution;
        }

        if (particle.position[i] > maxBoundary)
        {
            particle.position[i] = maxBoundary;
            particle.velocity[i] = -particle.velocity[i] * simulationData.restitution;
        }
    }
}

}
