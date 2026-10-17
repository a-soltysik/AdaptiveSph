#pragma once

#include <cuda_runtime_api.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "CudaError.cuh"
#include "DeviceSpan.cuh"

namespace sph::cuda
{

template <typename T>
class DeviceBuffer
{
public:
    DeviceBuffer() = default;

    static auto fromValue(const T& value, std::string_view context) -> DeviceBuffer
    {
        auto buffer = DeviceBuffer {1, context};
        buffer.copyValueFromHost(value, context);
        return buffer;
    }

    explicit DeviceBuffer(std::size_t size, std::string_view allocationContext)
    {
        allocate(size, allocationContext);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    auto operator=(const DeviceBuffer&) -> DeviceBuffer& = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : _data {std::exchange(other._data, nullptr)},
          _size {std::exchange(other._size, 0)}
    {
    }

    auto operator=(DeviceBuffer&& other) noexcept -> DeviceBuffer&
    {
        if (this != &other)
        {
            reset();
            _data = std::exchange(other._data, nullptr);
            _size = std::exchange(other._size, 0);
        }
        return *this;
    }

    ~DeviceBuffer()
    {
        reset();
    }

    void allocate(std::size_t size, std::string_view allocationContext)
    {
        if (size > std::numeric_limits<std::size_t>::max() / sizeof(T))
        {
            throw std::length_error {std::string {allocationContext} + " exceeds the addressable device-memory size"};
        }

        T* newData {};
        if (size != 0)
        {
            detail::checkCuda(cudaMalloc(reinterpret_cast<void**>(&newData), size * sizeof(T)), allocationContext);
        }

        reset();
        _data = newData;
        _size = size;
    }

    [[nodiscard]] auto data() noexcept -> T*
    {
        return _data;
    }

    [[nodiscard]] auto data() const noexcept -> const T*
    {
        return _data;
    }

    [[nodiscard]] auto size() const noexcept -> std::size_t
    {
        return _size;
    }

    [[nodiscard]] auto view() noexcept -> DeviceSpan<T>
    {
        return {_data, _size};
    }

    [[nodiscard]] auto view() const noexcept -> DeviceSpan<const T>
    {
        return {_data, _size};
    }

    void copyValueFromHost(const T& value, std::string_view context)
    {
        requireSingleValue(context);
        detail::checkCuda(cudaMemcpy(_data, &value, sizeof(T), cudaMemcpyHostToDevice), context);
    }

    [[nodiscard]] auto copyValueToHost(std::string_view context) const -> T
    {
        requireSingleValue(context);
        auto value = T {};
        detail::checkCuda(cudaMemcpy(&value, _data, sizeof(T), cudaMemcpyDeviceToHost), context);
        return value;
    }

private:
    void requireSingleValue(std::string_view context) const
    {
        if (_size != 1)
        {
            throw std::logic_error {std::string {context} + " requires a one-element device buffer"};
        }
    }

    void reset() noexcept
    {
        if (_data != nullptr)
        {
            detail::reportCudaCleanupError(cudaFree(_data), "DeviceBuffer: cudaFree");
            _data = nullptr;
            _size = 0;
        }
    }

    T* _data {};
    std::size_t _size {};
};

}
