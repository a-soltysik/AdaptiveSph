#pragma once

#include <cstdio>
#include <utility>

#if defined(WIN32)
#    include <Windows.h>
#else
#    include <unistd.h>

#    include <cerrno>
#endif

namespace sph::cuda
{

class ExternalMemoryHandle final
{
public:
#if defined(WIN32)
    using NativeHandle = void*;
#else
    using NativeHandle = int;
#endif

    explicit ExternalMemoryHandle(NativeHandle handle) noexcept
        : _handle {handle}
    {
    }

    ExternalMemoryHandle(const ExternalMemoryHandle&) = delete;
    auto operator=(const ExternalMemoryHandle&) -> ExternalMemoryHandle& = delete;

    ExternalMemoryHandle(ExternalMemoryHandle&& other) noexcept
        : _handle {other.release()}
    {
    }

    auto operator=(ExternalMemoryHandle&& other) noexcept -> ExternalMemoryHandle&
    {
        if (this != &other)
        {
            reset();
            _handle = other.release();
        }
        return *this;
    }

    ~ExternalMemoryHandle()
    {
        reset();
    }

    [[nodiscard]] auto get() const noexcept -> NativeHandle
    {
        return _handle;
    }

    [[nodiscard]] auto release() noexcept -> NativeHandle
    {
        const auto handle = _handle;
        _handle = invalidHandle;
        return handle;
    }

private:
#if defined(WIN32)
    static constexpr NativeHandle invalidHandle = nullptr;
#else
    static constexpr NativeHandle invalidHandle = -1;
#endif

    void reset() noexcept
    {
#if defined(WIN32)
        if (_handle != invalidHandle && CloseHandle(static_cast<HANDLE>(_handle)) == 0)
        {
            const auto error = GetLastError();
            std::fprintf(stderr,
                         "ExternalMemoryHandle: CloseHandle failed with system error %lu\n",
                         static_cast<unsigned long>(error));
        }
#else
        if (_handle != invalidHandle && close(_handle) != 0)
        {
            const auto error = errno;
            std::fprintf(stderr, "ExternalMemoryHandle: close file descriptor failed with system error %d\n", error);
        }
#endif
        _handle = invalidHandle;
    }

    NativeHandle _handle = invalidHandle;
};

}
