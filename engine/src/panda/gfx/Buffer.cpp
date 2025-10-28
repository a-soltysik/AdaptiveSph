// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include "panda/gfx/Buffer.h"

#include <cstddef>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "CommandBuffer.h"
#include "Device.h"
#include "panda/Logger.h"
#include "panda/utils/format/gfx/api/ResultFormatter.h"  // NOLINT(misc-include-cleaner)

namespace panda::gfx
{
auto Buffer::copy(const Buffer& src, const Buffer& dst) -> void
{
    const auto commandBuffer = CommandBuffer::beginSingleTimeCommandBuffer(src._device);

    const auto copyRegion = vk::BufferCopy {.size = src.size};
    commandBuffer.copyBuffer(src.buffer, dst.buffer, copyRegion);
    CommandBuffer::endSingleTimeCommandBuffer(src._device, commandBuffer);

    log::Info("Copied buffer [{}] to buffer [{}]", static_cast<void*>(src.buffer), static_cast<void*>(dst.buffer));
}

Buffer::Buffer(const Device& deviceRef,
               vk::DeviceSize instanceSize,
               size_t instanceCount,
               vk::BufferUsageFlags usage,
               vk::MemoryPropertyFlags properties,
               vk::DeviceSize minOffsetAlignment)
    : size {getAlignment(instanceSize, minOffsetAlignment) * instanceCount},
      buffer {createBuffer(deviceRef, size, usage)},
      memory {allocateMemory(deviceRef, buffer, properties)},
      _device {deviceRef},
      _minOffsetAlignment {minOffsetAlignment}
{
    expect(_device.logicalDevice.bindBufferMemory(buffer, memory, 0),
           vk::Result::eSuccess,
           "Failed to bind memory buffer");
    log::Info("Created new buffer [{}] with size: {}", static_cast<void*>(buffer), size);
}

Buffer::Buffer(const Device& deviceRef,
               vk::DeviceSize bufferSize,
               vk::BufferUsageFlags usage,
               vk::MemoryPropertyFlags properties)
    : Buffer {deviceRef, bufferSize, 1, usage, properties, 1}
{
}

Buffer::~Buffer() noexcept
{
    log::Info("Destroying buffer [{}]", static_cast<void*>(buffer));
    if (_mappedMemory != nullptr)
    {
        unmapWhole();
    }

    _device.logicalDevice.destroy(buffer);
    _device.logicalDevice.freeMemory(memory);
}

auto Buffer::flushWhole() const noexcept -> bool
{
    const auto mappedRange = vk::MappedMemoryRange {.memory = memory, .offset = 0, .size = size};
    return shouldBe(_device.logicalDevice.flushMappedMemoryRanges(mappedRange),
                    vk::Result::eSuccess,
                    "Failed flushing memory");
}

auto Buffer::flush(vk::DeviceSize dataSize, vk::DeviceSize offset) const noexcept -> bool
{
    const auto mappedRange = vk::MappedMemoryRange {.memory = memory, .offset = offset, .size = dataSize};
    return shouldBe(_device.logicalDevice.flushMappedMemoryRanges(mappedRange),
                    vk::Result::eSuccess,
                    "Failed flushing memory");
}

auto Buffer::mapWhole() noexcept -> void
{
    _mappedMemory = expect(_device.logicalDevice.mapMemory(memory, 0, size, {}),
                           vk::Result::eSuccess,
                           "Failed to map memory of vertex buffer");
}

auto Buffer::map(vk::DeviceSize dataSize, vk::DeviceSize offset) noexcept -> void
{
    _mappedMemory = expect(_device.logicalDevice.mapMemory(memory, offset, dataSize, {}),
                           vk::Result::eSuccess,
                           "Failed to map memory of vertex buffer");
}

auto Buffer::unmapWhole() noexcept -> void
{
    _device.logicalDevice.unmapMemory(memory);
    _mappedMemory = nullptr;
}

auto Buffer::createBuffer(const Device& device, vk::DeviceSize bufferSize, vk::BufferUsageFlags usage) -> vk::Buffer
{
    const auto bufferInfo =
        vk::BufferCreateInfo {.size = bufferSize, .usage = usage, .sharingMode = vk::SharingMode::eExclusive};
    return expect(device.logicalDevice.createBuffer(bufferInfo), vk::Result::eSuccess, "Failed to create buffer");
}

auto Buffer::allocateMemory(const Device& device, vk::Buffer buffer, vk::MemoryPropertyFlags properties)
    -> vk::DeviceMemory
{
    const auto memoryRequirements = device.logicalDevice.getBufferMemoryRequirements(buffer);
    const auto allocInfo = vk::MemoryAllocateInfo {
        .allocationSize = memoryRequirements.size,
        .memoryTypeIndex =
            expect(device.findMemoryType(memoryRequirements.memoryTypeBits, properties), "Failed to find memory type")};
    return expect(device.logicalDevice.allocateMemory(allocInfo),
                  vk::Result::eSuccess,
                  "Failed to allocated buffer memory");
}

auto Buffer::getAlignment(vk::DeviceSize instanceSize, vk::DeviceSize minOffsetAlignment) noexcept -> vk::DeviceSize
{
    return (instanceSize + minOffsetAlignment - 1) & ~(minOffsetAlignment - 1);
}

auto Buffer::getAlignment(vk::DeviceSize instanceSize) const noexcept -> vk::DeviceSize
{
    return getAlignment(instanceSize, _minOffsetAlignment);
}

auto Buffer::getCurrentOffset() const noexcept -> vk::DeviceSize
{
    return _currentOffset;
}

auto Buffer::getDescriptorInfo() const noexcept -> vk::DescriptorBufferInfo
{
    return getDescriptorInfoAt(size, 0);
}

auto Buffer::getDescriptorInfoAt(vk::DeviceSize dataSize, vk::DeviceSize offset) const noexcept
    -> vk::DescriptorBufferInfo
{
    return {
        .buffer = buffer,
        .offset = offset,
        .range = dataSize,
    };
}

}
