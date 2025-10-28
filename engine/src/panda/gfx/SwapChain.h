#pragma once

// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "panda/Common.h"
#include "panda/utils/Signals.h"

namespace panda
{
class Window;
}

namespace panda::gfx
{

class Device;

class SwapChain
{
public:
    SwapChain(const Device& device, const vk::SurfaceKHR& surface, const Window& window);
    PD_DELETE_ALL(SwapChain);
    ~SwapChain() noexcept;

    [[nodiscard]] auto getRenderPass() const noexcept -> const vk::RenderPass&;
    [[nodiscard]] auto getFrameBuffer(size_t index) const noexcept -> const vk::Framebuffer&;
    [[nodiscard]] auto getExtent() const noexcept -> const vk::Extent2D&;
    [[nodiscard]] auto getExtentAspectRatio() const noexcept -> float;
    [[nodiscard]] auto acquireNextImage() -> std::optional<uint32_t>;
    [[nodiscard]] auto imagesCount() const noexcept -> size_t;
    auto submitCommandBuffers(const vk::CommandBuffer& commandBuffer, uint32_t imageIndex) -> void;

private:
    [[nodiscard]] static auto createSwapChain(const vk::SurfaceKHR& surface,
                                              vk::Extent2D extent,
                                              const Device& device,
                                              const vk::SurfaceFormatKHR& surfaceFormat) -> vk::SwapchainKHR;
    [[nodiscard]] static auto chooseSwapSurfaceFormat(std::span<const vk::SurfaceFormatKHR> availableFormats) noexcept
        -> vk::SurfaceFormatKHR;
    [[nodiscard]] static auto choosePresentationMode(
        std::span<const vk::PresentModeKHR> availablePresentationModes) noexcept -> vk::PresentModeKHR;
    [[nodiscard]] static auto chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, const Window& window)
        -> vk::Extent2D;

    [[nodiscard]] static auto createImageViews(const std::vector<vk::Image>& swapChainImages,
                                               const vk::SurfaceFormatKHR& swapChainImageFormat,
                                               const Device& device) -> std::vector<vk::ImageView>;
    [[nodiscard]] static auto createRenderPass(const vk::SurfaceFormatKHR& imageFormat,
                                               const vk::SurfaceFormatKHR& depthFormat,
                                               const Device& device) -> vk::RenderPass;
    [[nodiscard]] static auto createFrameBuffers(const std::vector<vk::ImageView>& swapChainImageViews,
                                                 const std::vector<vk::ImageView>& depthImageViews,
                                                 const vk::RenderPass& renderPass,
                                                 vk::Extent2D swapChainExtent,
                                                 const Device& device) -> std::vector<vk::Framebuffer>;
    [[nodiscard]] static auto createDepthImages(const Device& device,
                                                vk::Extent2D swapChainExtent,
                                                size_t imagesCount,
                                                const vk::SurfaceFormatKHR& depthFormat) -> std::vector<vk::Image>;
    [[nodiscard]] static auto createDepthImageViews(const Device& device,
                                                    const std::vector<vk::Image>& depthImages,
                                                    size_t imagesCount,
                                                    const vk::SurfaceFormatKHR& depthFormat)
        -> std::vector<vk::ImageView>;
    [[nodiscard]] static auto createDepthImageMemories(const Device& device,
                                                       const std::vector<vk::Image>& depthImages,
                                                       size_t imagesCount) -> std::vector<vk::DeviceMemory>;
    [[nodiscard]] static auto findDepthFormat(const Device& device) -> vk::Format;

    auto createSyncObjects() -> void;
    auto cleanup() const -> void;
    auto recreate() -> void;

    const Device& _device;
    const Window& _window;
    const vk::SurfaceKHR& _surface;

    vk::Extent2D _swapChainExtent;
    vk::SurfaceFormatKHR _swapChainImageFormat;
    vk::SurfaceFormatKHR _swapChainDepthFormat;
    vk::SwapchainKHR _swapChain;
    std::vector<vk::Image> _swapChainImages;
    std::vector<vk::ImageView> _swapChainImageViews;
    std::vector<vk::Image> _depthImages;
    std::vector<vk::DeviceMemory> _depthImageMemories;
    std::vector<vk::ImageView> _depthImageViews;

    vk::RenderPass _renderPass;
    std::vector<vk::Framebuffer> _swapChainFrameBuffers;
    std::vector<vk::Semaphore> _imageAvailableSemaphores;
    std::vector<vk::Semaphore> _renderFinishedSemaphores;
    std::vector<vk::Fence> _inFlightFences;
    std::vector<vk::Fence*> _imagesInFlight;

    utils::signals::FrameBufferResized::ReceiverT _frameBufferResizeReceiver;
    uint32_t _currentFrame = 0;
    bool _frameBufferResized = false;
};

}
