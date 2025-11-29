#pragma once

#include <memory>

#include "../Api.cuh"

namespace sph::cuda
{

class SPH_CUDA_API ImportedMemory
{
public:
    ImportedMemory() = default;
    ImportedMemory(const ImportedMemory&) = delete;
    ImportedMemory(ImportedMemory&&) = delete;

    auto operator=(const ImportedMemory&) -> ImportedMemory& = delete;
    auto operator=(ImportedMemory&&) -> ImportedMemory& = delete;
    virtual ~ImportedMemory() = default;
};

#if defined(WIN32)
SPH_CUDA_API auto importBuffer(void* handle, size_t size) -> std::unique_ptr<ImportedMemory>;
#else
SPH_CUDA_API auto importBuffer(int handle, size_t size) -> std::unique_ptr<ImportedMemory>;
#endif

}
