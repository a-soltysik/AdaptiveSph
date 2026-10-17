#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <cstddef>
#include <cuda/memory/ImportedMemory.cuh>
#include <memory>
#include <utility>

#include "../utils/CudaError.cuh"
#include "ImportedParticleMemory.cuh"

namespace sph::cuda
{

ImportedParticleMemory::~ImportedParticleMemory()
{
    if (_data != nullptr)
    {
        detail::reportCudaCleanupError(cudaFree(_data), "ImportedParticleMemory: unmap imported buffer");
    }
    if (_externalMemory != nullptr)
    {
        detail::reportCudaCleanupError(cudaDestroyExternalMemory(_externalMemory),
                                       "ImportedParticleMemory: destroy imported memory");
    }
}
#if defined(WIN32)
ImportedParticleMemory::ImportedParticleMemory(ExternalMemoryHandle handle, size_t size)
    : _size {size}
{
    cudaExternalMemoryHandleDesc handleDesc {};
    handleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    handleDesc.size = size;
    handleDesc.handle.win32.handle = handle.get();

    detail::checkCuda(cudaImportExternalMemory(&_externalMemory, &handleDesc),
                      "ImportedParticleMemory: import Win32 external memory");
    mapBuffer();
}
#else
ImportedParticleMemory::ImportedParticleMemory(ExternalMemoryHandle handle, size_t size)
    : _size {size}
{
    cudaExternalMemoryHandleDesc handleDesc {};
    handleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    handleDesc.size = size;
    handleDesc.handle.fd = handle.get();

    detail::checkCuda(cudaImportExternalMemory(&_externalMemory, &handleDesc),
                      "ImportedParticleMemory: import file-descriptor external memory");
    static_cast<void>(handle.release());
    mapBuffer();
}
#endif

void ImportedParticleMemory::mapBuffer()
{
    cudaExternalMemoryBufferDesc bufferDesc {};
    bufferDesc.size = _size;
    bufferDesc.offset = 0;

    try
    {
        detail::checkCuda(cudaExternalMemoryGetMappedBuffer(&_data, _externalMemory, &bufferDesc),
                          "ImportedParticleMemory: map external buffer");
    }
    catch (...)
    {
        detail::reportCudaCleanupError(cudaDestroyExternalMemory(_externalMemory),
                                       "ImportedParticleMemory: roll back external-memory import");
        _externalMemory = {};
        throw;
    }
}

#if defined(WIN32)
auto importBuffer(void* handle, size_t size) -> std::unique_ptr<ImportedMemory>
{
    auto ownedHandle = ExternalMemoryHandle {handle};
    return std::make_unique<ImportedParticleMemory>(std::move(ownedHandle), size);
}
#else
auto importBuffer(int handle, size_t size) -> std::unique_ptr<ImportedMemory>
{
    auto ownedHandle = ExternalMemoryHandle {handle};
    return std::make_unique<ImportedParticleMemory>(std::move(ownedHandle), size);
}
#endif

}
