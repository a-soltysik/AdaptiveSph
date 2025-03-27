#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <cstddef>
#include <glm/ext/vector_float3.hpp>
#include <memory>
#include <vector>

#include "ImportedParticleMemory.cuh"
#include "Kernel.cuh"
#include "SphSimulation.cuh"
#include "cuda/ImportedMemory.cuh"
#include "cuda/Simulation.cuh"

namespace sph::cuda
{

SphSimulation::SphSimulation(const SimulationData& data,
                             const std::vector<glm::vec3>& positions,
                             const ImportedMemory& memory)
    : _particleBuffer {dynamic_cast<const ImportedParticleMemory&>(memory)},
      _simulationData {data},
      _particleCount {positions.size()}
{
    std::vector<ParticleData> particles(positions.size());

    for (size_t i = 0; i < positions.size(); i++)
    {
        particles[i].position = positions[i];
        particles[i].force = {0, 0.0005, 0};
        particles[i].mass = getParticleMassBasedOnDensity(data.density);
    }

    cudaMemcpy(_particleBuffer.getParticles(),
               particles.data(),
               particles.size() * sizeof(ParticleData),
               cudaMemcpyHostToDevice);
}

auto SphSimulation::getParticleMassBasedOnDensity(float density) -> float
{
    return density / 50;
}

void SphSimulation::update(FrameData data)
{
    updatePositions<<<(_particleCount + 255) / 256, 256>>>(_particleBuffer.getParticles(),
                                                           _particleCount,
                                                           data.deltaTime);
    handleCollisions<<<(_particleCount + 255) / 256, 256>>>(_particleBuffer.getParticles(),
                                                            _particleCount,
                                                            _simulationData);
    cudaDeviceSynchronize();
}

auto createSimulation(const Simulation::SimulationData& data,
                      const std::vector<glm::vec3>& positions,
                      const ImportedMemory& memory) -> std::unique_ptr<Simulation>
{
    return std::make_unique<SphSimulation>(data, positions, memory);
}

}
