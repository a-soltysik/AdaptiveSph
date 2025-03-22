#pragma once
// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "Device.h"

namespace panda::gfx::vulkan
{

class CommandBuffer
{
public:
    [[nodiscard]] static auto beginSingleTimeCommandBuffer(const Device& device) noexcept -> vk::CommandBuffer;
    static auto endSingleTimeCommandBuffer(const Device& device, vk::CommandBuffer buffer) noexcept -> void;
};

}
