// clang-format off
#include "cuda/ImportedMemory.cuh"
#include "panda/utils/Assert.h"
// clang-format on

#include "panda/gfx/vulkan/SharedBuffer.hpp"

#include <vulkan/vulkan.h>  // NOLINT(misc-include-cleaner)

#include <cstddef>
#include <memory>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "panda/Logger.h"
#include "panda/gfx/vulkan/Device.h"
#include "panda/utils/Utils.h"
#include "panda/utils/format/gfx/api/vulkan/ResultFormatter.h"  // NOLINT(misc-include-cleaner)

namespace panda::gfx::vulkan
{

SharedBuffer::SharedBuffer(const Device& deviceRef,
                           vk::DeviceSize instanceSize,
                           size_t instanceCount,
                           vk::MemoryPropertyFlags properties,
                           vk::DeviceSize minOffsetAlignment)
    : size {getAlignment(instanceSize, minOffsetAlignment) * instanceCount},
      buffer {createBuffer(deviceRef, size)},
      memory {allocateMemory(deviceRef, buffer, properties)},
      _device {deviceRef},
      _minOffsetAlignment {minOffsetAlignment}
{
    expect(_device.logicalDevice.bindBufferMemory(buffer, memory, 0),
           vk::Result::eSuccess,
           "Failed to bind memory buffer");
    _importedMemory = sph::cuda::importBuffer(getBufferHandle(), size);
    log::Info("Created new buffer [{}] with size: {}", static_cast<void*>(buffer), size);

    _bufferDestructor = std::make_unique<utils::ScopeGuard>([this] {
        _device.logicalDevice.destroy(buffer);
        _device.logicalDevice.freeMemory(memory);
    });
}

SharedBuffer::SharedBuffer(const Device& deviceRef, vk::DeviceSize bufferSize, vk::MemoryPropertyFlags properties)
    : SharedBuffer {deviceRef, bufferSize, 1, properties, 1}
{
}

SharedBuffer::~SharedBuffer()
{
    log::Info("Destroying shared buffer");
    shouldBe(_device.logicalDevice.waitIdle(), vk::Result::eSuccess, "Failed to wait idle device");
}

[[nodiscard]] auto SharedBuffer::getImportedMemory() const -> const sph::cuda::ImportedMemory&
{
    return *_importedMemory;
}

auto SharedBuffer::createBuffer(const Device& device, vk::DeviceSize bufferSize) -> vk::Buffer
{
#if defined(WIN32)
    static constexpr auto externalInfo =
        vk::ExternalMemoryBufferCreateInfo {vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32};
#else
    static constexpr auto externalInfo =
        vk::ExternalMemoryBufferCreateInfo {vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd};
#endif

    const auto bufferInfo = vk::BufferCreateInfo {{},
                                                  bufferSize,
                                                  vk::BufferUsageFlagBits::eStorageBuffer,
                                                  vk::SharingMode::eExclusive,
                                                  {},
                                                  {},
                                                  &externalInfo};

    return expect(device.logicalDevice.createBuffer(bufferInfo), vk::Result::eSuccess, "Failed to create buffer");
}

#if defined(WIN32)
auto SharedBuffer::getBufferHandle() const -> void*
{
    const auto getInfo = vk::MemoryGetWin32HandleInfoKHR {memory, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32};
    return expect(_device.logicalDevice.getMemoryWin32HandleKHR(getInfo),
                  vk::Result::eSuccess,
                  "Failed to get memory handle");
}

#else

auto SharedBuffer::getBufferHandle() const -> int
{
    const auto getInfo = vk::MemoryGetFdInfoKHR {memory, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd};
    return expect(_device.logicalDevice.getMemoryFdKHR(getInfo), vk::Result::eSuccess, "Failed to get memory handle");
}
#endif

auto SharedBuffer::allocateMemory(const Device& device, vk::Buffer buffer, vk::MemoryPropertyFlags properties)
    -> vk::DeviceMemory
{
#if defined(WIN32)
    static constexpr auto exportAllocationInfo =
        vk::ExportMemoryAllocateInfo {vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32};
#else
    static constexpr auto exportAllocationInfo =
        vk::ExportMemoryAllocateInfo {vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd};
#endif

    const auto memoryRequirements = device.logicalDevice.getBufferMemoryRequirements(buffer);
    const auto allocInfo = vk::MemoryAllocateInfo {
        memoryRequirements.size,
        expect(device.findMemoryType(memoryRequirements.memoryTypeBits, properties), "Failed to find memory type"),
        &exportAllocationInfo};

    return expect(device.logicalDevice.allocateMemory(allocInfo),
                  vk::Result::eSuccess,
                  "Failed to allocated buffer memory");
}

auto SharedBuffer::getAlignment(vk::DeviceSize instanceSize, vk::DeviceSize minOffsetAlignment) noexcept
    -> vk::DeviceSize
{
    return (instanceSize + minOffsetAlignment - 1) & ~(minOffsetAlignment - 1);
}

auto SharedBuffer::getAlignment(vk::DeviceSize instanceSize) const noexcept -> vk::DeviceSize
{
    return getAlignment(instanceSize, _minOffsetAlignment);
}

auto SharedBuffer::getCurrentOffset() const noexcept -> vk::DeviceSize
{
    return _currentOffset;
}

auto SharedBuffer::getDescriptorInfo() const noexcept -> vk::DescriptorBufferInfo
{
    return getDescriptorInfoAt(size, 0);
}

auto SharedBuffer::getDescriptorInfoAt(vk::DeviceSize dataSize, vk::DeviceSize offset) const noexcept
    -> vk::DescriptorBufferInfo
{
    return {
        buffer,
        offset,
        dataSize,
    };
}

}
