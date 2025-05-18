#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace sph::cuda
{

template <typename T>
class Memory
{
public:
    explicit Memory(size_t size)
        : _size(size),
          _data(nullptr)
    {
        if (size > 0)
        {
            cudaMalloc(&_data, size * sizeof(T));
        }
    }

    ~Memory()
    {
        if (_data)
        {
            cudaFree(_data);
        }
    }

    Memory(const Memory&) = delete;
    auto operator=(const Memory&) -> Memory& = delete;

    Memory(Memory&& other) noexcept
        : _size(other._size),
          _data(other._data)
    {
        other._data = nullptr;
        other._size = 0;
    }

    auto operator=(Memory&& other) noexcept -> Memory&
    {
        if (this != &other)
        {
            if (_data)
            {
                cudaFree(_data);
            }
            _data = other._data;
            _size = other._size;
            other._data = nullptr;
            other._size = 0;
        }
        return *this;
    }

    [[nodiscard]] auto get() const -> T*
    {
        return _data;
    }

    [[nodiscard]] auto size() const -> size_t
    {
        return _size;
    }

private:
    size_t _size;
    T* _data;
};

}
