#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cuda/ImportedMemory.cuh>

namespace sph::cuda
{

class ImportedParticleMemory : public ImportedMemory
{
public:
    ImportedParticleMemory(const ImportedParticleMemory&) = delete;
    auto operator=(const ImportedParticleMemory&) -> ImportedParticleMemory& = delete;
    ImportedParticleMemory(ImportedParticleMemory&&) = delete;
    ImportedParticleMemory& operator=(ImportedParticleMemory&&) = delete;
    ~ImportedParticleMemory() override;

#if defined(WIN32)
    ImportedParticleMemory(void* handle, size_t size);
#else
    ImportedParticleMemory(int handle, size_t size);
#endif

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
    cudaExternalMemory_t _externalMemory {};
    void* _data {};
    size_t _size = 0;
};
}
