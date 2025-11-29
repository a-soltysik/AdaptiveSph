#pragma once

// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include <../../../../cuda/include/cuda/memory/ImportedMemory.cuh>
#include <cstddef>
#include <memory>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "Device.h"
#include "panda/Common.h"
#include "panda/utils/Utils.h"

namespace panda::gfx
{
class SharedBuffer
{
public:
    SharedBuffer(const Device& deviceRef, vk::DeviceSize bufferSize, vk::MemoryPropertyFlags properties);

    SharedBuffer(const Device& deviceRef,
                 vk::DeviceSize instanceSize,
                 size_t instanceCount,
                 vk::MemoryPropertyFlags properties,
                 vk::DeviceSize minOffsetAlignment = 1);

    PD_DELETE_ALL(SharedBuffer);

    ~SharedBuffer();

    [[nodiscard]] auto getAlignment(vk::DeviceSize instanceSize) const noexcept -> vk::DeviceSize;
    [[nodiscard]] auto getCurrentOffset() const noexcept -> vk::DeviceSize;
    [[nodiscard]] auto getDescriptorInfo() const noexcept -> vk::DescriptorBufferInfo;
    [[nodiscard]] auto getDescriptorInfoAt(vk::DeviceSize dataSize, vk::DeviceSize offset) const noexcept
        -> vk::DescriptorBufferInfo;

#if defined(WIN32)
    [[nodiscard]] auto getBufferHandle() const -> void*;
#else
    [[nodiscard]] auto getBufferHandle() const -> int;
#endif

    [[nodiscard]] auto getImportedMemory() const -> const sph::cuda::ImportedMemory&;

    const vk::DeviceSize size;
    const vk::Buffer buffer;
    const vk::DeviceMemory memory;

private:
    [[nodiscard]] static auto createBuffer(const Device& device, vk::DeviceSize bufferSize) -> vk::Buffer;
    [[nodiscard]] static auto allocateMemory(const Device& device,
                                             vk::Buffer buffer,
                                             vk::MemoryPropertyFlags properties) -> vk::DeviceMemory;
    [[nodiscard]] static auto getAlignment(vk::DeviceSize instanceSize, vk::DeviceSize minOffsetAlignment) noexcept
        -> vk::DeviceSize;

    const Device& _device;
    const vk::DeviceSize _minOffsetAlignment;
    vk::DeviceSize _currentOffset = 0;

    std::unique_ptr<utils::ScopeGuard> _bufferDestructor;
    std::unique_ptr<sph::cuda::ImportedMemory> _importedMemory;
};
}
