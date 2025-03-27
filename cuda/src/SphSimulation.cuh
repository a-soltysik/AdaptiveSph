#pragma once
#include <cstddef>
#include <glm/ext/vector_float3.hpp>
#include <vector>

#include "ImportedParticleMemory.cuh"
#include "cuda/ImportedMemory.cuh"
#include "cuda/Simulation.cuh"

namespace sph::cuda
{

class SphSimulation : public Simulation
{
public:
    SphSimulation(const SimulationData& data, const std::vector<glm::vec3>& positions, const ImportedMemory& memory);
    void update(FrameData data) override;

private:
    static auto getParticleMassBasedOnDensity(float density) -> float;

    const ImportedParticleMemory& _particleBuffer;
    SimulationData _simulationData;
    size_t _particleCount = 0;
};
}
