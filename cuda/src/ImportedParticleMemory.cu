#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <cstddef>
#include <memory>

#include "ImportedParticleMemory.cuh"
#include "cuda/ImportedMemory.cuh"
#include "cuda/Simulation.cuh"

namespace sph::cuda
{

ImportedParticleMemory::~ImportedParticleMemory()
{
    cudaFree(_particles);
    cudaDestroyExternalMemory(_externalMemory);
}
#if defined(WIN32)
ImportedParticleMemory::ImportedParticleMemory(void* handle, size_t size)
    : _size {size}
{
    cudaExternalMemoryHandleDesc handleDesc {};
    handleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    handleDesc.size = size;
    handleDesc.handle.win32.handle = handle;

    cudaImportExternalMemory(&_externalMemory, &handleDesc);

    cudaExternalMemoryBufferDesc bufferDesc {};
    bufferDesc.size = size;
    bufferDesc.offset = 0;

    cudaExternalMemoryGetMappedBuffer(reinterpret_cast<void**>(&_particles), _externalMemory, &bufferDesc);
}
#else
ImportedParticleMemory::ImportedParticleMemory(int handle, size_t size)
    : _size {size}
{
    cudaExternalMemoryHandleDesc handleDesc {};
    handleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    handleDesc.size = size;
    handleDesc.handle.fd = handle;

    cudaImportExternalMemory(&_externalMemory, &handleDesc);

    cudaExternalMemoryBufferDesc bufferDesc {};
    bufferDesc.size = size;
    bufferDesc.offset = 0;

    cudaExternalMemoryGetMappedBuffer(reinterpret_cast<void**>(&_particles), _externalMemory, &bufferDesc);
}
#endif

auto ImportedParticleMemory::getParticles() const -> ParticleData*
{
    return _particles;
}

auto ImportedParticleMemory::getSize() const -> size_t
{
    return _size;
}

auto ImportedParticleMemory::getMaxParticleCount() const -> size_t
{
    return _size / sizeof(ParticleData);
}

#if defined(WIN32)
auto importBuffer(void* handle, size_t size) -> std::unique_ptr<ImportedMemory>
{
    return std::make_unique<ImportedParticleMemory>(handle, size);
}
#else
auto importBuffer(int handle, size_t size) -> std::unique_ptr<ImportedMemory>
{
    return std::make_unique<ImportedParticleMemory>(handle, size);
}
#endif

}
