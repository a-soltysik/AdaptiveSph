#pragma once

#include <cstddef>

#if defined(__CUDACC__)
#    define SPH_CUDA_HOST_DEVICE __host__ __device__
#else
#    define SPH_CUDA_HOST_DEVICE
#endif

namespace sph::cuda
{

template <typename T>
class DeviceSpan
{
public:
    SPH_CUDA_HOST_DEVICE constexpr DeviceSpan() noexcept = default;

    SPH_CUDA_HOST_DEVICE constexpr DeviceSpan(T* data, std::size_t size) noexcept
        : _data {data},
          _size {size}
    {
    }

    [[nodiscard]] SPH_CUDA_HOST_DEVICE constexpr auto data() const noexcept -> T*
    {
        return _data;
    }

    [[nodiscard]] SPH_CUDA_HOST_DEVICE constexpr auto size() const noexcept -> std::size_t
    {
        return _size;
    }

    [[nodiscard]] SPH_CUDA_HOST_DEVICE constexpr auto empty() const noexcept -> bool
    {
        return _size == 0;
    }

    [[nodiscard]] SPH_CUDA_HOST_DEVICE constexpr auto operator[](std::size_t index) const noexcept -> T&
    {
        return _data[index];
    }

private:
    T* _data {};
    std::size_t _size {};
};

}

#undef SPH_CUDA_HOST_DEVICE
