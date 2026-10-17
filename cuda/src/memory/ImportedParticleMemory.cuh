#pragma once

#include <cuda_runtime.h>
#include <driver_types.h>

#include <cstddef>
#include <cuda/memory/ImportedMemory.cuh>

#include "ExternalMemoryHandle.hpp"

namespace sph::cuda
{

class ImportedParticleMemory : public ImportedMemory
{
public:
    ImportedParticleMemory(const ImportedParticleMemory&) = delete;
    auto operator=(const ImportedParticleMemory&) -> ImportedParticleMemory& = delete;
    ImportedParticleMemory(ImportedParticleMemory&&) = delete;
    auto operator=(ImportedParticleMemory&&) -> ImportedParticleMemory& = delete;
    ~ImportedParticleMemory() override;

    ImportedParticleMemory(ExternalMemoryHandle handle, size_t size);

    template <typename T>
    [[nodiscard]] auto getData() const -> T*
    {
        return static_cast<T*>(_data);
    }

    [[nodiscard]] auto getSize() const -> size_t
    {
        return _size;
    }

    template <typename T>
    [[nodiscard]] auto getMaxDataCount() const -> size_t
    {
        return _size / sizeof(T);
    }

private:
    void mapBuffer();

    cudaExternalMemory_t _externalMemory {};
    void* _data {};
    size_t _size = 0;
};
}
