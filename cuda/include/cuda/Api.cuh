#pragma once

#include <array>
#include <cstdint>
#include <optional>

#ifdef _WIN32
#    ifdef ASPH_CUDA_EXPORTS
#        define SPH_CUDA_API __declspec(dllexport)
#    else
#        define SPH_CUDA_API __declspec(dllimport)
#    endif
#else
#    define SPH_CUDA_API
#endif

namespace sph::cuda
{
SPH_CUDA_API auto selectDeviceForUuid(const std::array<std::uint8_t, 16>& uuid) -> std::optional<int>;
}
