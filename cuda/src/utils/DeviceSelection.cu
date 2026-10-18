#include <cuda_runtime_api.h>

#include <array>
#include <cstring>
#include <cuda/Api.cuh>
#include <optional>

#include "CudaError.cuh"

namespace sph::cuda
{

auto selectDeviceForUuid(const std::array<std::uint8_t, 16>& uuid) -> std::optional<int>
{
    auto deviceCount = int {};
    detail::checkCuda(cudaGetDeviceCount(&deviceCount), "CUDA device selection: enumerate devices");

    for (auto deviceIndex = int {}; deviceIndex < deviceCount; ++deviceIndex)
    {
        auto properties = cudaDeviceProp {};
        detail::checkCuda(cudaGetDeviceProperties(&properties, deviceIndex),
                          "CUDA device selection: query device properties");

        if (std::memcmp(uuid.data(), properties.uuid.bytes, uuid.size()) == 0)
        {
            detail::checkCuda(cudaSetDevice(deviceIndex), "CUDA device selection: select UUID-matched device");
            return deviceIndex;
        }
    }

    return std::nullopt;
}

}
