#pragma once
#include <cstddef>

#include "ImportedParticleMemory.cuh"
#include "cuda/Simulation.cuh"

namespace sph::cuda
{

class SphSimulation : public Simulation
{
public:
    SphSimulation(const SimulationData& data, const ImportedMemory& memory);
    void update(FrameData data) override;

private:
    const ImportedParticleMemory& _particleBuffer;
    size_t _particleCount = 0;
};
}
