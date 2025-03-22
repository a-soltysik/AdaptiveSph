#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <cstddef>
#include <vector>

#include "ImportedParticleMemory.cuh"
#include "Kernel.cuh"
#include "SphSimulation.cuh"

namespace sph::cuda
{

SphSimulation::SphSimulation(const SimulationData& data, const ImportedMemory& memory)
    : _particleBuffer {dynamic_cast<const ImportedParticleMemory&>(memory)}
{
    std::vector<ParticleData> particles(data.positions.size());

    for (size_t i = 0; i < data.positions.size(); i++)
    {
        particles[i].position = {data.positions[i].x, data.positions[i].y, data.positions[i].z};
        particles[i].force = {0, 0.0005, 0};
        particles[i].mass = 0.0125;
    }

    cudaMemcpy(_particleBuffer.getParticles(),
               particles.data(),
               particles.size() * sizeof(ParticleData),
               cudaMemcpyHostToDevice);
    _particleCount = data.positions.size();
}

void SphSimulation::update(FrameData data)
{
    updatePositions<<<(_particleCount + 255) / 256, 256>>>(_particleBuffer.getParticles(),
                                                           _particleCount,
                                                           data.deltaTime);
    cudaDeviceSynchronize();
}

std::unique_ptr<Simulation> createSimulation(const Simulation::SimulationData& data, const ImportedMemory& memory)
{
    return std::make_unique<SphSimulation>(data, memory);
}

}
