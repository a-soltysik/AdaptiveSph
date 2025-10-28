// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include "panda/gfx/object/Texture.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <glm/ext/vector_float4.hpp>
#include <memory>
#include <span>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "panda/gfx/Buffer.h"
#include "panda/gfx/CommandBuffer.h"
#include "panda/gfx/Context.h"
#include "panda/gfx/Device.h"
#include "panda/utils/format/gfx/api/ResultFormatter.h"  // NOLINT(misc-include-cleaner)

namespace panda::gfx
{

namespace
{
auto copyBufferToImage(const Device& device, const Buffer& buffer, vk::Image image, size_t width, size_t height) -> void
{
    const auto commandBuffer = CommandBuffer::beginSingleTimeCommandBuffer(device);

    const auto region = vk::BufferImageCopy {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                             .mipLevel = 0,
                             .baseArrayLayer = 0,
                             .layerCount = 1},
        .imageOffset = {.x = 0, .y = 0, .z = 0},
        .imageExtent = {.width = static_cast<uint32_t>(width), .height = static_cast<uint32_t>(height), .depth = 1}
    };
    commandBuffer.copyBufferToImage(buffer.buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
    CommandBuffer::endSingleTimeCommandBuffer(device, commandBuffer);
}

auto transitionImageLayout(const Device& device, vk::Image image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout)
    -> void
{
    const auto commandBuffer = CommandBuffer::beginSingleTimeCommandBuffer(device);

    auto barrier = vk::ImageMemoryBarrier {
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = image,
        .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                             .baseMipLevel = 0,
                             .levelCount = 1,
                             .baseArrayLayer = 0,
                             .layerCount = 1}
    };

    auto sourceStage = vk::PipelineStageFlags {};
    auto destinationStage = vk::PipelineStageFlags {};

    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal)
    {
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;
    }
    else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
    {
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        sourceStage = vk::PipelineStageFlagBits::eTransfer;
        destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    }

    commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, {}, barrier);

    CommandBuffer::endSingleTimeCommandBuffer(device, commandBuffer);
}

auto createTextureImageView(const Device& device, vk::Image image)
{
    const auto viewInfo = vk::ImageViewCreateInfo {
        .image = image,
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eR8G8B8A8Srgb,
        .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                             .baseMipLevel = 0,
                             .levelCount = 1,
                             .baseArrayLayer = 0,
                             .layerCount = 1}
    };

    return expect(device.logicalDevice.createImageView(viewInfo), vk::Result::eSuccess, "Failed to create image view");
}

auto createTextureSampler(const Device& device) -> vk::Sampler
{
    const auto samplerInfo =
        vk::SamplerCreateInfo {.magFilter = vk::Filter::eLinear,
                               .minFilter = vk::Filter::eLinear,
                               .mipmapMode = vk::SamplerMipmapMode::eLinear,
                               .addressModeU = vk::SamplerAddressMode::eRepeat,
                               .addressModeV = vk::SamplerAddressMode::eRepeat,
                               .addressModeW = vk::SamplerAddressMode::eRepeat,
                               .mipLodBias = 0.F,
                               .anisotropyEnable = vk::True,
                               .maxAnisotropy = device.physicalDevice.getProperties().limits.maxSamplerAnisotropy,
                               .compareEnable = vk::False,
                               .compareOp = vk::CompareOp::eAlways,
                               .minLod = 0.F,
                               .maxLod = 0.F,
                               .borderColor = vk::BorderColor::eIntOpaqueBlack,
                               .unnormalizedCoordinates = vk::False};

    return expect(device.logicalDevice.createSampler(samplerInfo), vk::Result::eSuccess, "Failed to create sampler");
}
}

Texture::~Texture()
{
    _context.getDevice().logicalDevice.destroy(_sampler);
    _context.getDevice().logicalDevice.destroy(_imageView);
    _context.getDevice().logicalDevice.destroy(_image);
    _context.getDevice().logicalDevice.free(_imageMemory);
}

auto Texture::getDescriptorImageInfo() const noexcept -> vk::DescriptorImageInfo
{
    return vk::DescriptorImageInfo {.sampler = _sampler,
                                    .imageView = _imageView,
                                    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
}

auto Texture::getDefaultTexture(const Context& context, glm::vec4 color) -> std::unique_ptr<Texture>
{
    static constexpr auto max = int32_t {255};
    return std::make_unique<Texture>(
        context,
        std::array {static_cast<uint8_t>(std::clamp(static_cast<int32_t>(max * color.x), 0, max)),
                    static_cast<uint8_t>(std::clamp(static_cast<int32_t>(max * color.y), 0, max)),
                    static_cast<uint8_t>(std::clamp(static_cast<int32_t>(max * color.z), 0, max)),
                    static_cast<uint8_t>(std::clamp(static_cast<int32_t>(max * color.w), 0, max))},
        1,
        1);
}

Texture::Texture(const Context& context, std::span<const uint8_t> data, size_t width, size_t height)
    : _context {context}
{
    load(data, width, height);
}

auto Texture::load(std::span<const uint8_t> data, size_t width, size_t height) -> void
{
    const auto imageSize = width * height * 4;
    auto stagingBuffer = Buffer {_context.getDevice(),
                                 static_cast<vk::DeviceSize>(imageSize),
                                 vk::BufferUsageFlagBits::eTransferSrc,
                                 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent};

    stagingBuffer.mapWhole();
    stagingBuffer.write(data.data(), imageSize);
    stagingBuffer.unmapWhole();

    const auto imageInfo = vk::ImageCreateInfo {
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR8G8B8A8Srgb,
        .extent = {.width = static_cast<uint32_t>(width), .height = static_cast<uint32_t>(height), .depth = 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined
    };

    _image = expect(_context.getDevice().logicalDevice.createImage(imageInfo),
                    vk::Result::eSuccess,
                    "Failed to create image");

    const auto memoryRequirements = _context.getDevice().logicalDevice.getImageMemoryRequirements(_image);

    const auto allocationInfo = vk::MemoryAllocateInfo {
        .allocationSize = memoryRequirements.size,
        .memoryTypeIndex = expect(_context.getDevice().findMemoryType(memoryRequirements.memoryTypeBits,
                                                                      vk::MemoryPropertyFlagBits::eDeviceLocal),
                                  "Can't find device local memory"),
    };

    _imageMemory = expect(_context.getDevice().logicalDevice.allocateMemory(allocationInfo),
                          vk::Result::eSuccess,
                          "Failed to allocate memory");
    expect(_context.getDevice().logicalDevice.bindImageMemory(_image, _imageMemory, 0),
           vk::Result::eSuccess,
           "Failed to bind image memory");

    transitionImageLayout(_context.getDevice(),
                          _image,
                          vk::ImageLayout::eUndefined,
                          vk::ImageLayout::eTransferDstOptimal);

    copyBufferToImage(_context.getDevice(), stagingBuffer, _image, width, height);
    transitionImageLayout(_context.getDevice(),
                          _image,
                          vk::ImageLayout::eTransferDstOptimal,
                          vk::ImageLayout::eShaderReadOnlyOptimal);

    _imageView = createTextureImageView(_context.getDevice(), _image);
    _sampler = createTextureSampler(_context.getDevice());
}

}
