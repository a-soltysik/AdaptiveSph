// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include "panda/gfx/Renderer.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "SwapChain.h"
#include "panda/gfx/Context.h"
#include "panda/gfx/Device.h"
#include "panda/utils/format/gfx/api/ResultFormatter.h"  // NOLINT(misc-include-cleaner)

namespace panda::gfx
{

Renderer::Renderer(const Window& window, const Device& device, const vk::SurfaceKHR& surface)
    : _device {device},
      _swapChain {std::make_unique<SwapChain>(device, surface, window)},
      _commandBuffers {createCommandBuffers()}
{
}

Renderer::~Renderer() noexcept
{
    _device.logicalDevice.freeCommandBuffers(_device.commandPool, _commandBuffers);
}

auto Renderer::beginFrame() -> vk::CommandBuffer
{
    expectNot(_isFrameStarted, "Can't begin frame when already began");

    const auto imageIndex = _swapChain->acquireNextImage();
    if (!imageIndex.has_value())
    {
        return nullptr;
    }
    _currentImageIndex = imageIndex.value();
    _isFrameStarted = true;
    const auto commandBuffer = getCurrentCommandBuffer();
    static constexpr auto beginInfo = vk::CommandBufferBeginInfo {};
    expect(commandBuffer.begin(beginInfo), vk::Result::eSuccess, "Can't begin commandBuffer");
    return commandBuffer;
}

auto Renderer::endFrame() -> void
{
    expect(_isFrameStarted, "Can't end frame which isn't began");
    expect(getCurrentCommandBuffer().end(), vk::Result::eSuccess, "Can't end command buffer");
    _swapChain->submitCommandBuffers(getCurrentCommandBuffer(), _currentImageIndex);

    _isFrameStarted = false;
    _currentFrameIndex = (_currentFrameIndex + 1) % Context::maxFramesInFlight;
}

auto Renderer::beginSwapChainRenderPass() const -> void
{
    expect(_isFrameStarted, "Can't begin render pass when frame is not began");
    static constexpr auto clearColor =
        vk::ClearValue {vk::ClearColorValue {.float32 {std::array {0.08F, 0.08F, 0.1F, 1.F}}}};
    static constexpr auto depthStencil = vk::ClearValue {
        .depthStencil = vk::ClearDepthStencilValue {.depth = 1.F, .stencil = 0}
    };
    static constexpr auto clearValues = std::array {clearColor, depthStencil};
    const auto renderPassBeginInfo = vk::RenderPassBeginInfo {
        .renderPass = _swapChain->getRenderPass(),
        .framebuffer = _swapChain->getFrameBuffer(_currentImageIndex),
        .renderArea = {.offset = {.x = 0, .y = 0}, .extent = _swapChain->getExtent()},
        .clearValueCount = clearValues.size(),
        .pClearValues = clearValues.data(),
    };

    const auto commandBuffer = getCurrentCommandBuffer();

    commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

    const auto viewport = vk::Viewport {.x = 0.F,
                                        .y = 0.F,
                                        .width = static_cast<float>(_swapChain->getExtent().width),
                                        .height = static_cast<float>(_swapChain->getExtent().height),
                                        .minDepth = 0.F,
                                        .maxDepth = 1.F};
    commandBuffer.setViewport(0, viewport);

    const auto scissor = vk::Rect2D {
        .offset = {.x = 0, .y = 0},
        .extent = _swapChain->getExtent()
    };

    commandBuffer.setScissor(0, scissor);
}

auto Renderer::endSwapChainRenderPass() const -> void
{
    expect(_isFrameStarted, "Can't end render pass when frame is not began");
    getCurrentCommandBuffer().endRenderPass();
}

auto Renderer::isFrameInProgress() const noexcept -> bool
{
    return _isFrameStarted;
}

auto Renderer::getCurrentCommandBuffer() const noexcept -> const vk::CommandBuffer&
{
    expect(_isFrameStarted, "Can't get command buffer when frame not in progress");
    return _commandBuffers[_currentFrameIndex];
}

auto Renderer::getSwapChainRenderPass() const noexcept -> const vk::RenderPass&
{
    return _swapChain->getRenderPass();
}

auto Renderer::createCommandBuffers() const -> std::vector<vk::CommandBuffer>
{
    const auto allocationInfo =
        vk::CommandBufferAllocateInfo {.commandPool = _device.commandPool,
                                       .level = vk::CommandBufferLevel::ePrimary,
                                       .commandBufferCount = static_cast<uint32_t>(_swapChain->imagesCount())};
    return expect(_device.logicalDevice.allocateCommandBuffers(allocationInfo),
                  vk::Result::eSuccess,
                  "Can't allocate command buffer");
}

auto Renderer::getFrameIndex() const noexcept -> uint32_t
{
    expect(_isFrameStarted, "Can't get frame index which is not in progress");
    return _currentFrameIndex;
}

auto Renderer::getAspectRatio() const noexcept -> float
{
    return _swapChain->getExtentAspectRatio();
}

}
