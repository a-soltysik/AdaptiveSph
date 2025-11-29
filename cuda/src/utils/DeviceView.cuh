#pragma once
#include <driver_types.h>

namespace sph::cuda
{

template <typename T>
class DeviceView
{
public:
    static auto fromDevice(const T* devicePtr) -> DeviceView
    {
        return DeviceView {devicePtr};
    }

    DeviceView() = delete;
    DeviceView(DeviceView&& value) noexcept = default;
    auto operator=(DeviceView&& value) noexcept -> DeviceView& = default;

    DeviceView(const DeviceView&) = delete;
    auto operator=(const DeviceView&) -> DeviceView& = delete;
    ~DeviceView() = default;

    auto toHost() const -> T
    {
        auto hostValue = T {};
        cudaMemcpy(&hostValue, _devicePtr, sizeof(T), cudaMemcpyDeviceToHost);
        return hostValue;
    }

    auto getDevicePtr() const -> const T*
    {
        return _devicePtr;
    }

private:
    explicit DeviceView(const T* devicePtr)
        : _devicePtr {devicePtr}
    {
    }

    const T* _devicePtr;
};

}