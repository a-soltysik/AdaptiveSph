#pragma once

#include <memory>

#include "cuda/Api.cuh"

namespace sph::cuda
{

class SPH_CUDA_API ImportedSemaphore
{
public:
    virtual ~ImportedSemaphore() = default;

    virtual auto wait() const -> void = 0;
    virtual auto signal() const -> void = 0;
};

#if defined(WIN32)
SPH_CUDA_API auto importSemaphore(void* handle) -> std::unique_ptr<ImportedSemaphore>;
#else
SPH_CUDA_API auto importSemaphore(int handle) -> std::unique_ptr<ImportedSemaphore>;
#endif

}
