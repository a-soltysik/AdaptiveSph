// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include "SwapChain.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "Device.h"
#include "panda/Logger.h"
#include "panda/Window.h"
#include "panda/gfx/Context.h"
#include "panda/utils/Signals.h"
#include "panda/utils/format/gfx/api/ResultFormatter.h"  // NOLINT(misc-include-cleaner)

namespace panda::gfx
{

SwapChain::SwapChain(const Device& device, const vk::SurfaceKHR& surface, const Window& window)
    : _device {device},
      _window {window},
      _surface {surface},
      _swapChainExtent {chooseSwapExtent(_device.querySwapChainSupport().capabilities, _window)},
      _swapChainImageFormat {chooseSwapSurfaceFormat(_device.querySwapChainSupport().formats)},
      _swapChainDepthFormat {.format = findDepthFormat(_device)},
      _swapChain {createSwapChain(_surface, _swapChainExtent, _device, _swapChainImageFormat)},
      _swapChainImages {expect(
          _device.logicalDevice.getSwapchainImagesKHR(_swapChain), vk::Result::eSuccess, "Can't get swapchain images")},
      _swapChainImageViews {createImageViews(_swapChainImages, _swapChainImageFormat, _device)},
      _depthImages {createDepthImages(_device, _swapChainExtent, _swapChainImages.size(), _swapChainDepthFormat)},
      _depthImageMemories {createDepthImageMemories(_device, _depthImages, _swapChainImages.size())},
      _depthImageViews {createDepthImageViews(_device, _depthImages, _swapChainImages.size(), _swapChainDepthFormat)},
      _renderPass {createRenderPass(_swapChainImageFormat, _swapChainDepthFormat, _device)},
      _swapChainFrameBuffers {
          createFrameBuffers(_swapChainImageViews, _depthImageViews, _renderPass, _swapChainExtent, _device)},
      _frameBufferResizeReceiver {utils::signals::frameBufferResized.connect([this](auto) noexcept {
          log::Debug("Received framebuffer resized notif");
          _frameBufferResized = true;
      })}
{
    createSyncObjects();
}

SwapChain::~SwapChain() noexcept
{
    cleanup();

    _device.logicalDevice.destroy(_renderPass);
    for (const auto semaphore : _renderFinishedSemaphores)
    {
        _device.logicalDevice.destroy(semaphore);
    }
    for (const auto semaphore : _imageAvailableSemaphores)
    {
        _device.logicalDevice.destroy(semaphore);
    }
    for (const auto fence : _inFlightFences)
    {
        _device.logicalDevice.destroy(fence);
    }
}

auto SwapChain::choosePresentationMode(std::span<const vk::PresentModeKHR> availablePresentationModes) noexcept
    -> vk::PresentModeKHR
{
    const auto it = std::ranges::find(availablePresentationModes, vk::PresentModeKHR::eMailbox);

    if (it == availablePresentationModes.end())
    {
        log::Warning("Choosing default mode: Fifo");
        return vk::PresentModeKHR::eFifo;
    }
    return *it;
}

auto SwapChain::chooseSwapSurfaceFormat(std::span<const vk::SurfaceFormatKHR> availableFormats) noexcept
    -> vk::SurfaceFormatKHR
{
    const auto it = std::ranges::find_if(availableFormats, [](const auto& availableFormat) noexcept {
        return availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
               availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
    });

    if (it == availableFormats.end())
    {
        log::Warning("Choosing default format");
        return availableFormats.front();
    }
    return *it;
}

auto SwapChain::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, const Window& window) -> vk::Extent2D
{
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) [[likely]]
    {
        return capabilities.currentExtent;
    }

    const auto size = window.getSize();

    return {
        .width = std::clamp<uint32_t>(size.x, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
        .height = std::clamp<uint32_t>(size.y, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)};
}

auto SwapChain::createImageViews(const std::vector<vk::Image>& swapChainImages,
                                 const vk::SurfaceFormatKHR& swapChainImageFormat,
                                 const Device& device) -> std::vector<vk::ImageView>
{
    auto imageViews = std::vector<vk::ImageView> {};
    imageViews.reserve(swapChainImages.size());

    for (const auto& image : swapChainImages)
    {
        static constexpr auto subResourceRange =
            vk::ImageSubresourceRange {.aspectMask = vk::ImageAspectFlagBits::eColor,
                                       .baseMipLevel = 0,
                                       .levelCount = 1,
                                       .baseArrayLayer = 0,
                                       .layerCount = 1};
        const auto createInfo = vk::ImageViewCreateInfo {.image = image,
                                                         .viewType = vk::ImageViewType::e2D,
                                                         .format = swapChainImageFormat.format,
                                                         .subresourceRange = subResourceRange};

        imageViews.push_back(
            expect(device.logicalDevice.createImageView(createInfo), vk::Result::eSuccess, "Can't create image view"));
    }

    return imageViews;
}

auto SwapChain::createRenderPass(const vk::SurfaceFormatKHR& imageFormat,
                                 const vk::SurfaceFormatKHR& depthFormat,
                                 const Device& device) -> vk::RenderPass
{
    const auto depthAttachment =
        vk::AttachmentDescription {.format = depthFormat.format,
                                   .samples = vk::SampleCountFlagBits::e1,
                                   .loadOp = vk::AttachmentLoadOp::eClear,
                                   .storeOp = vk::AttachmentStoreOp::eDontCare,
                                   .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
                                   .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
                                   .initialLayout = vk::ImageLayout::eUndefined,
                                   .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal};

    static constexpr auto depthAttachmentRef =
        vk::AttachmentReference {.attachment = 1, .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal};

    const auto colorAttachment = vk::AttachmentDescription {.format = imageFormat.format,
                                                            .samples = vk::SampleCountFlagBits::e1,
                                                            .loadOp = vk::AttachmentLoadOp::eClear,
                                                            .storeOp = vk::AttachmentStoreOp::eStore,
                                                            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
                                                            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
                                                            .initialLayout = vk::ImageLayout::eUndefined,
                                                            .finalLayout = vk::ImageLayout::ePresentSrcKHR};

    static constexpr auto colorAttachmentRef =
        vk::AttachmentReference {.attachment = 0, .layout = vk::ImageLayout::eColorAttachmentOptimal};

    static constexpr auto subpass = vk::SubpassDescription {.pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
                                                            .colorAttachmentCount = 1,
                                                            .pColorAttachments = &colorAttachmentRef,
                                                            .pDepthStencilAttachment = &depthAttachmentRef};

    static constexpr auto dependency = vk::SubpassDependency {
        .srcSubpass = vk::SubpassExternal,
        .dstSubpass = 0,
        .srcStageMask =
            vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
        .dstStageMask =
            vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
        .srcAccessMask = vk::AccessFlagBits::eNone,
        .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite};

    const auto attachments = std::array {colorAttachment, depthAttachment};

    const auto renderPassInfo = vk::RenderPassCreateInfo {.attachmentCount = attachments.size(),
                                                          .pAttachments = attachments.data(),
                                                          .subpassCount = 1,
                                                          .pSubpasses = &subpass,
                                                          .dependencyCount = 1,
                                                          .pDependencies = &dependency};

    return expect(device.logicalDevice.createRenderPass(renderPassInfo),
                  vk::Result::eSuccess,
                  "Can't create render pass");
}

auto SwapChain::createFrameBuffers(const std::vector<vk::ImageView>& swapChainImageViews,
                                   const std::vector<vk::ImageView>& depthImageViews,
                                   const vk::RenderPass& renderPass,
                                   vk::Extent2D swapChainExtent,
                                   const Device& device) -> std::vector<vk::Framebuffer>
{
    expect(swapChainImageViews.size() == depthImageViews.size(),
           "Swap chain image views count is different than depth image views count");
    auto result = std::vector<vk::Framebuffer> {};
    result.reserve(swapChainImageViews.size());

    for (auto i = size_t {}; i < swapChainImageViews.size(); i++)
    {
        const auto attachments = std::array {swapChainImageViews[i], depthImageViews[i]};
        const auto frameBufferInfo = vk::FramebufferCreateInfo {.renderPass = renderPass,
                                                                .attachmentCount = attachments.size(),
                                                                .pAttachments = attachments.data(),
                                                                .width = swapChainExtent.width,
                                                                .height = swapChainExtent.height,
                                                                .layers = 1};
        result.push_back(expect(device.logicalDevice.createFramebuffer(frameBufferInfo),
                                vk::Result::eSuccess,
                                "Can't create framebuffer"));
    }
    return result;
}

auto SwapChain::createSyncObjects() -> void
{
    static constexpr auto semaphoreInfo = vk::SemaphoreCreateInfo {};
    static constexpr auto fenceInfo = vk::FenceCreateInfo {.flags = vk::FenceCreateFlagBits::eSignaled};

    _imageAvailableSemaphores.reserve(Context::maxFramesInFlight);
    _renderFinishedSemaphores.reserve(imagesCount());
    _inFlightFences.reserve(Context::maxFramesInFlight);
    _imagesInFlight.resize(imagesCount());

    for (auto i = size_t {}; i < Context::maxFramesInFlight; i++)
    {
        _imageAvailableSemaphores.push_back(expect(_device.logicalDevice.createSemaphore(semaphoreInfo),
                                                   vk::Result::eSuccess,
                                                   "Failed to create imageAvailable semaphore"));
    }

    for (auto i = size_t {}; i < imagesCount(); i++)
    {
        _renderFinishedSemaphores.push_back(expect(_device.logicalDevice.createSemaphore(semaphoreInfo),
                                                   vk::Result::eSuccess,
                                                   "Failed to create renderFinished semaphore"));
    }

    for (auto i = size_t {}; i < Context::maxFramesInFlight; i++)
    {
        _inFlightFences.push_back(
            expect(_device.logicalDevice.createFence(fenceInfo), vk::Result::eSuccess, "Failed to create fence"));
    }
}

auto SwapChain::recreate() -> void
{
    log::Info("Starting to recreate swapchain");
    shouldBe(_device.logicalDevice.waitIdle(), vk::Result::eSuccess, "Wait idle didn't succeed");

    cleanup();
    const auto swapChainSupport = _device.querySwapChainSupport();
    _swapChainExtent = chooseSwapExtent(swapChainSupport.capabilities, _window);

    const auto oldImageFormat = _swapChainImageFormat;
    _swapChainImageFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    shouldBe(oldImageFormat != _swapChainImageFormat, "Image format has changed!");

    const auto oldDepthFormat = _swapChainDepthFormat;
    _swapChainDepthFormat.format = findDepthFormat(_device);
    shouldBe(oldDepthFormat != _swapChainDepthFormat, "Depth format has changed!");

    _swapChain = createSwapChain(_surface, _swapChainExtent, _device, _swapChainImageFormat);
    _swapChainImages = expect(_device.logicalDevice.getSwapchainImagesKHR(_swapChain),
                              vk::Result::eSuccess,
                              "Can't get swapchain images");
    _swapChainImageViews = createImageViews(_swapChainImages, _swapChainImageFormat, _device);
    _depthImages = createDepthImages(_device, _swapChainExtent, _swapChainImages.size(), _swapChainDepthFormat);
    _depthImageMemories = createDepthImageMemories(_device, _depthImages, _swapChainImages.size());
    _depthImageViews = createDepthImageViews(_device, _depthImages, _swapChainImages.size(), _swapChainDepthFormat);
    _swapChainFrameBuffers =
        createFrameBuffers(_swapChainImageViews, _depthImageViews, _renderPass, _swapChainExtent, _device);

    log::Info("Swapchain recreated");
}

auto SwapChain::cleanup() const -> void
{
    for (const auto framebuffer : _swapChainFrameBuffers)
    {
        _device.logicalDevice.destroy(framebuffer);
    }
    for (const auto imageView : _swapChainImageViews)
    {
        _device.logicalDevice.destroy(imageView);
    }
    for (const auto image : _depthImages)
    {
        _device.logicalDevice.destroy(image);
    }
    for (const auto imageView : _depthImageViews)
    {
        _device.logicalDevice.destroy(imageView);
    }
    for (const auto imageMemory : _depthImageMemories)
    {
        _device.logicalDevice.free(imageMemory);
    }
    _device.logicalDevice.destroy(_swapChain);
}

auto SwapChain::createSwapChain(const vk::SurfaceKHR& surface,
                                const vk::Extent2D extent,
                                const Device& device,
                                const vk::SurfaceFormatKHR& surfaceFormat) -> vk::SwapchainKHR
{
    const auto swapChainSupport = device.querySwapChainSupport();
    const auto presentationMode = choosePresentationMode(swapChainSupport.presentationModes);

    auto imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
    {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    auto createInfo = vk::SwapchainCreateInfoKHR {.surface = surface,
                                                  .minImageCount = imageCount,
                                                  .imageFormat = surfaceFormat.format,
                                                  .imageColorSpace = surfaceFormat.colorSpace,
                                                  .imageExtent = extent,
                                                  .imageArrayLayers = 1,
                                                  .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
                                                  .preTransform = swapChainSupport.capabilities.currentTransform,
                                                  .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
                                                  .presentMode = presentationMode,
                                                  .clipped = vk::True};

    if (device.queueFamilies.graphicsFamily != device.queueFamilies.presentationFamily)
    {
        const auto indicesArray =
            std::array {device.queueFamilies.graphicsFamily, device.queueFamilies.presentationFamily};
        createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
        createInfo.setQueueFamilyIndices(indicesArray);
    }
    else
    {
        createInfo.imageSharingMode = vk::SharingMode::eExclusive;
    }

    return expect(device.logicalDevice.createSwapchainKHR(createInfo), vk::Result::eSuccess, "Can't create swapchain");
}

auto SwapChain::getRenderPass() const noexcept -> const vk::RenderPass&
{
    return _renderPass;
}

auto SwapChain::getFrameBuffer(size_t index) const noexcept -> const vk::Framebuffer&
{
    return _swapChainFrameBuffers[index];
}

auto SwapChain::getExtent() const noexcept -> const vk::Extent2D&
{
    return _swapChainExtent;
}

auto SwapChain::findDepthFormat(const Device& device) -> vk::Format
{
    return expect(device.findSupportedFormat(
                      std::array {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
                      vk::ImageTiling::eOptimal,
                      vk::FormatFeatureFlagBits::eDepthStencilAttachment),
                  "Failed to find supported format");
}

auto SwapChain::acquireNextImage() -> std::optional<uint32_t>
{
    if (_frameBufferResized) [[unlikely]]
    {
        _frameBufferResized = false;
        recreate();
        return {};
    }

    shouldBe(_device.logicalDevice.waitForFences(_inFlightFences[_currentFrame],
                                                 vk::True,
                                                 std::numeric_limits<uint64_t>::max()),
             vk::Result::eSuccess,
             "Waiting for the fences didn't succeed");

    const auto imageIndex = _device.logicalDevice.acquireNextImageKHR(_swapChain,
                                                                      std::numeric_limits<uint64_t>::max(),
                                                                      _imageAvailableSemaphores[_currentFrame]);

    if (imageIndex.result == vk::Result::eErrorOutOfDateKHR || imageIndex.result == vk::Result::eSuboptimalKHR)
        [[unlikely]]
    {
        recreate();
        return {};
    }
    if (imageIndex.result != vk::Result::eSuccess) [[unlikely]]
    {
        panic("Failed to acquire swap chain image");
    }

    return imageIndex.value;
}

auto SwapChain::submitCommandBuffers(const vk::CommandBuffer& commandBuffer, uint32_t imageIndex) -> void
{
    if (_imagesInFlight[imageIndex] != nullptr) [[likely]]
    {
        shouldBe(_device.logicalDevice.waitForFences(*_imagesInFlight[imageIndex],
                                                     vk::True,
                                                     std::numeric_limits<uint64_t>::max()),
                 vk::Result::eSuccess,
                 "Failed to wait for _imagesInFlight fence");
    }
    _imagesInFlight[imageIndex] = &_inFlightFences[_currentFrame];

    static constexpr auto waitStages =
        std::array {vk::PipelineStageFlags {vk::PipelineStageFlagBits::eColorAttachmentOutput}};
    const auto submitInfo = vk::SubmitInfo {.waitSemaphoreCount = 1,
                                            .pWaitSemaphores = &_imageAvailableSemaphores[_currentFrame],
                                            .pWaitDstStageMask = waitStages.data(),
                                            .commandBufferCount = 1,
                                            .pCommandBuffers = &commandBuffer,
                                            .signalSemaphoreCount = 1,
                                            .pSignalSemaphores = &_renderFinishedSemaphores[imageIndex]};

    shouldBe(_device.logicalDevice.resetFences(_inFlightFences[_currentFrame]),
             vk::Result::eSuccess,
             "Failed to Reset inFlight fence");
    shouldBe(_device.graphicsQueue.submit(submitInfo, _inFlightFences[_currentFrame]),
             vk::Result::eSuccess,
             "Submitting the graphics queue didn't succeeded");

    const auto presentInfo = vk::PresentInfoKHR {.waitSemaphoreCount = 1,
                                                 .pWaitSemaphores = &_renderFinishedSemaphores[imageIndex],
                                                 .swapchainCount = 1,
                                                 .pSwapchains = &_swapChain,
                                                 .pImageIndices = &imageIndex};

    const auto presentationResult = _device.presentationQueue.presentKHR(presentInfo);

    if (presentationResult == vk::Result::eErrorOutOfDateKHR || presentationResult == vk::Result::eSuboptimalKHR)
        [[unlikely]]
    {
        recreate();
    }
    else if (presentationResult != vk::Result::eSuccess) [[unlikely]]
    {
        log::Warning("Presenting the queue didn't succeeded: {}", presentationResult);
    }

    _currentFrame = (_currentFrame + 1) % Context::maxFramesInFlight;
}

auto SwapChain::imagesCount() const noexcept -> size_t
{
    return _swapChainImages.size();
}

auto SwapChain::createDepthImages(const Device& device,
                                  vk::Extent2D swapChainExtent,
                                  size_t imagesCount,
                                  const vk::SurfaceFormatKHR& depthFormat) -> std::vector<vk::Image>
{
    auto depthImages = std::vector<vk::Image> {};
    depthImages.reserve(imagesCount);

    for (auto i = size_t {}; i < imagesCount; i++)
    {
        const auto imageInfo = vk::ImageCreateInfo {
            .imageType = vk::ImageType::e2D,
            .format = depthFormat.format,
            .extent = {.width = swapChainExtent.width, .height = swapChainExtent.height, .depth = 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
            .sharingMode = vk::SharingMode::eExclusive
        };
        depthImages.push_back(
            expect(device.logicalDevice.createImage(imageInfo), vk::Result::eSuccess, "Failed to create depth image"));
    }
    return depthImages;
}

auto SwapChain::createDepthImageViews(const Device& device,
                                      const std::vector<vk::Image>& depthImages,
                                      size_t imagesCount,
                                      const vk::SurfaceFormatKHR& depthFormat) -> std::vector<vk::ImageView>
{
    auto depthImageViews = std::vector<vk::ImageView> {};
    depthImageViews.reserve(imagesCount);

    for (auto i = size_t {}; i < imagesCount; i++)
    {
        const auto viewInfo = vk::ImageViewCreateInfo {
            .image = depthImages[i],
            .viewType = vk::ImageViewType::e2D,
            .format = depthFormat.format,
            .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eDepth,
                                 .baseMipLevel = 0,
                                 .levelCount = 1,
                                 .baseArrayLayer = 0,
                                 .layerCount = 1}
        };
        depthImageViews.push_back(expect(device.logicalDevice.createImageView(viewInfo),
                                         vk::Result::eSuccess,
                                         "Failed to create depth image view"));
    }
    return depthImageViews;
}

auto SwapChain::createDepthImageMemories(const Device& device,
                                         const std::vector<vk::Image>& depthImages,
                                         size_t imagesCount) -> std::vector<vk::DeviceMemory>
{
    auto depthImageMemories = std::vector<vk::DeviceMemory> {};
    depthImageMemories.reserve(imagesCount);

    for (auto i = size_t {}; i < imagesCount; i++)
    {
        const auto memoryRequirements = device.logicalDevice.getImageMemoryRequirements(depthImages[i]);
        const auto allocInfo = vk::MemoryAllocateInfo {
            .allocationSize = memoryRequirements.size,
            .memoryTypeIndex = expect(
                device.findMemoryType(memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
                "Failed to find memory type")};
        depthImageMemories.push_back(expect(device.logicalDevice.allocateMemory(allocInfo),
                                            vk::Result::eSuccess,
                                            "Failed to allocate depth image memory"));
        expect(device.logicalDevice.bindImageMemory(depthImages[i], depthImageMemories[i], 0),
               vk::Result::eSuccess,
               "Failed to bind depth image memory");
    }
    return depthImageMemories;
}

auto SwapChain::getExtentAspectRatio() const noexcept -> float
{
    return static_cast<float>(_swapChainExtent.width) / static_cast<float>(_swapChainExtent.height);
}

}