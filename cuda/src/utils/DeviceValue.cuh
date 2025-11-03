#pragma once
#include <driver_types.h>

#include <memory>

namespace sph::cuda
{

template <typename T>
class DeviceValue
{
public:
    static auto fromDevice(T* devicePtr) -> DeviceValue
    {
        return DeviceValue {devicePtr};
    }

    static auto fromHost(const T& value) -> DeviceValue
    {
        T* devicePtr {};
        cudaMalloc(&devicePtr, sizeof(T));
        cudaMemcpy(devicePtr, &value, sizeof(T), cudaMemcpyHostToDevice);

        return DeviceValue {devicePtr};
    }

    DeviceValue() = delete;
    DeviceValue(DeviceValue&& value) noexcept = default;
    auto operator=(DeviceValue&& value) noexcept -> DeviceValue& = default;

    DeviceValue(const DeviceValue&) = delete;
    auto operator=(const DeviceValue&) -> DeviceValue& = delete;
    ~DeviceValue() = default;

    auto operator=(const T& value) -> DeviceValue&
    {
        cudaMemcpy(_devicePtr.get(), &value, sizeof(T), cudaMemcpyHostToDevice);

        return *this;
    }

    auto toHost() -> T
    {
        auto hostValue = T {};
        cudaMemcpy(&hostValue, _devicePtr.get(), sizeof(T), cudaMemcpyDeviceToHost);
        return hostValue;
    }

    auto getDevicePtr() -> T*
    {
        return _devicePtr.get();
    }

private:
    struct Deleter
    {
        void operator()(T* ptr) const noexcept
        {
            cudaFree(ptr);
        }
    };

    explicit DeviceValue(T* devicePtr)
        : _devicePtr {devicePtr, Deleter {}}
    {
    }

    std::unique_ptr<T, Deleter> _devicePtr;
};

}