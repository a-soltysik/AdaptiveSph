#include <cuda_runtime_api.h>

#include <cuda/memory/ImportedSemaphore.cuh>
#include <memory>
#include <utility>

#include "../utils/CudaError.cuh"
#include "ExternalMemoryHandle.hpp"

namespace sph::cuda
{

namespace
{
class ImportedSemaphoreImpl final : public ImportedSemaphore
{
public:
#if defined(WIN32)
    explicit ImportedSemaphoreImpl(ExternalMemoryHandle handle)
    {
        cudaExternalSemaphoreHandleDesc handleDesc {};
        handleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
        handleDesc.handle.win32.handle = handle.get();

        detail::checkCuda(cudaImportExternalSemaphore(&_semaphore, &handleDesc),
                          "ImportedSemaphore: import Win32 external semaphore");
    }
#else
    explicit ImportedSemaphoreImpl(ExternalMemoryHandle handle)
    {
        cudaExternalSemaphoreHandleDesc handleDesc {};
        handleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
        handleDesc.handle.fd = handle.get();

        detail::checkCuda(cudaImportExternalSemaphore(&_semaphore, &handleDesc),
                          "ImportedSemaphore: import file-descriptor external semaphore");
        static_cast<void>(handle.release());
    }
#endif

    ~ImportedSemaphoreImpl() override
    {
        if (_semaphore != nullptr)
        {
            detail::reportCudaCleanupError(cudaDestroyExternalSemaphore(_semaphore),
                                           "ImportedSemaphore: destroy external semaphore");
        }
    }

    auto wait() const -> void override
    {
        const auto waitParams = cudaExternalSemaphoreWaitParams {};
        detail::checkCuda(cudaWaitExternalSemaphoresAsync(&_semaphore, &waitParams, 1, nullptr),
                          "ImportedSemaphore: enqueue external semaphore wait");
    }

    auto signal() const -> void override
    {
        const auto signalParams = cudaExternalSemaphoreSignalParams {};
        detail::checkCuda(cudaSignalExternalSemaphoresAsync(&_semaphore, &signalParams, 1, nullptr),
                          "ImportedSemaphore: enqueue external semaphore signal");
    }

    cudaExternalSemaphore_t _semaphore {};
};
}

#if defined(WIN32)
auto importSemaphore(void* handle) -> std::unique_ptr<ImportedSemaphore>
{
    auto ownedHandle = ExternalMemoryHandle {handle};
    return std::unique_ptr<ImportedSemaphore> {new ImportedSemaphoreImpl {std::move(ownedHandle)}};
}
#else
auto importSemaphore(int handle) -> std::unique_ptr<ImportedSemaphore>
{
    auto ownedHandle = ExternalMemoryHandle {handle};
    return std::unique_ptr<ImportedSemaphore> {new ImportedSemaphoreImpl {std::move(ownedHandle)}};
}
#endif

}
