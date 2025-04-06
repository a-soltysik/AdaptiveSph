#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "cuda/ImportedMemory.cuh"
#include "cuda/Simulation.cuh"

namespace sph::cuda
{

class ImportedParticleMemory : public ImportedMemory
{
public:
    ~ImportedParticleMemory() override;
#if defined(WIN32)
    ImportedParticleMemory(void* handle, size_t size);
#else
    ImportedParticleMemory(int handle, size_t size);
#endif

    [[nodiscard]] auto getParticles() const -> ParticleData*;

    [[nodiscard]] auto getSize() const -> size_t;

    [[nodiscard]] auto getMaxParticleCount() const -> size_t;

private:
    cudaExternalMemory_t _externalMemory {};
    ParticleData* _particles {};
    size_t _size = 0;
};
}
