#include <cuda_runtime_api.h>

#include <cstddef>

#include "cuda/kernel.cuh"

namespace sph::cuda
{

struct Particle
{
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 force;
    float density;
    float pressure;
    float mass;
};

std::vector<glm::vec3> gPositions;
Particle* gParticles;
size_t gNumParticles;

void initialize(const SimulationData& data)
{
    std::vector<Particle> particles(data.positions.size());

    for (size_t i = 0; i < data.positions.size(); i++)
    {
        particles[i].position = {data.positions[i].x, data.positions[i].y, data.positions[i].z};
        particles[i].force = {0, 1, 0};
        particles[i].mass = 1;
    }
    cudaMalloc(reinterpret_cast<void**>(&gParticles), data.positions.size() * sizeof(Particle));

    cudaMemcpy(gParticles, particles.data(), particles.size() * sizeof(Particle), cudaMemcpyHostToDevice);
    gNumParticles = data.positions.size();
}

__global__ void updatePositions(Particle* particles, int numObjects, float dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numObjects)
    {
        particles[idx].velocity += particles[idx].force / particles[idx].mass * dt;
        particles[idx].position += particles[idx].velocity * dt;
    }
}

void update(FrameData data)
{
    updatePositions<<<(gNumParticles + 255) / 256, 256>>>(gParticles, gNumParticles, data.deltaTime);
    cudaDeviceSynchronize();
}

__global__ void extractPositions(Particle* particles, glm::vec3* positions, int numObjects)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numObjects)
    {
        positions[idx] = particles[idx].position;
    }
}

void getUpdatedPositions(std::vector<glm::vec3*>& objects)
{
    gPositions.resize(objects.size());

    glm::vec3* devPositions;
    cudaMalloc((void**) &devPositions, objects.size() * sizeof(float3));

    extractPositions<<<(objects.size() + 255) / 256, 256>>>(gParticles, devPositions, objects.size());
    cudaMemcpy(gPositions.data(), devPositions, objects.size() * sizeof(float3), cudaMemcpyDeviceToHost);

    cudaFree(devPositions);

    for (size_t i = 0; i < objects.size(); i++)
    {
        objects[i]->x = gPositions[i].x;
        objects[i]->y = gPositions[i].y;
        objects[i]->z = gPositions[i].z;
    }
}

}
