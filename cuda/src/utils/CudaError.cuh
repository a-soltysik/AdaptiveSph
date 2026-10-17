#pragma once

#include <cuda_runtime_api.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sph::cuda::detail
{

inline auto cudaErrorMessage(cudaError_t status, std::string_view operation) -> std::string
{
    const auto* name = cudaGetErrorName(status);
    const auto* description = cudaGetErrorString(status);

    return std::string {operation} + " failed with " + (name != nullptr ? name : "unknown CUDA error") + ": " +
           (description != nullptr ? description : "no error description");
}

inline void checkCuda(cudaError_t status, std::string_view operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error {cudaErrorMessage(status, operation)};
    }
}

inline void checkCudaLastError(std::string_view operation)
{
    checkCuda(cudaGetLastError(), operation);
}

inline void reportCudaCleanupError(cudaError_t status, std::string_view operation) noexcept
{
    if (status == cudaSuccess)
    {
        return;
    }

    const auto* description = cudaGetErrorString(status);
    std::fprintf(stderr,
                 "CUDA cleanup: %.*s failed with %s (%d)\n",
                 static_cast<int>(operation.size()),
                 operation.data(),
                 description != nullptr ? description : "unknown CUDA error",
                 static_cast<int>(status));
}

}
