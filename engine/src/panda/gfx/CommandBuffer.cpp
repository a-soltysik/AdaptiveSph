// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include "panda/gfx/CommandBuffer.h"

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "Device.h"
#include "panda/utils/format/gfx/api/ResultFormatter.h"  // NOLINT(misc-include-cleaner)

namespace panda::gfx
{

auto CommandBuffer::beginSingleTimeCommandBuffer(const Device& device) noexcept -> vk::CommandBuffer
{
    const auto allocationInfo = vk::CommandBufferAllocateInfo {.commandPool = device.commandPool,
                                                               .level = vk::CommandBufferLevel::ePrimary,
                                                               .commandBufferCount = 1};
    const auto commandBuffer = expect(device.logicalDevice.allocateCommandBuffers(allocationInfo),
                                      vk::Result::eSuccess,
                                      "Can't allocate command buffer");
    static constexpr auto beginInfo =
        vk::CommandBufferBeginInfo {.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
    expect(commandBuffer.front().begin(beginInfo), vk::Result::eSuccess, "Couldn't begin command buffer");
    return commandBuffer.front();
}

auto CommandBuffer::endSingleTimeCommandBuffer(const Device& device, vk::CommandBuffer buffer) noexcept -> void
{
    expect(buffer.end(), vk::Result::eSuccess, "Couldn't end command buffer");

    const auto submitInfo = vk::SubmitInfo {.commandBufferCount = 1, .pCommandBuffers = &buffer};
    expect(device.graphicsQueue.submit(submitInfo), vk::Result::eSuccess, "Couldn't submit graphics queue");
    shouldBe(device.graphicsQueue.waitIdle(), vk::Result::eSuccess, "Couldn't wait idle on graphics queue");
    device.logicalDevice.free(device.commandPool, buffer);
}
}
