// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include "FluidParticleRenderSystem.h"

#include <cstddef>
#include <cuda/Simulation.cuh>
#include <filesystem>
#include <memory>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "panda/gfx/Buffer.h"
#include "panda/gfx/Descriptor.h"
#include "panda/gfx/Device.h"
#include "panda/gfx/FrameInfo.h"
#include "panda/gfx/Pipeline.h"
#include "panda/gfx/Scene.h"
#include "panda/internal/config.h"
#include "panda/utils/format/gfx/api/ResultFormatter.h"  // NOLINT(misc-include-cleaner)

namespace panda::gfx
{
FluidParticleRenderSystem::FluidParticleRenderSystem(const Device& device,
                                                     vk::RenderPass renderPass,
                                                     size_t particleCount)
    : _device {device},
      _descriptorLayout {DescriptorSetLayout::Builder(_device)
                             .addBinding(0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex)
                             .addBinding(1, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eFragment)
                             .addBinding(2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eVertex)
                             .addBinding(3, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eVertex)
                             .addBinding(4, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eVertex)
                             .build(vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR)},
      _pipelineLayout {createPipelineLayout(_device, _descriptorLayout->getDescriptorSetLayout())},
      _pipeline {createPipeline(_device, renderPass, _pipelineLayout)},
      _particleBuffer {
          .positions =
              createSharedBufferFromPointerType<decltype(sph::cuda::FluidParticlesDataImportedBuffer::positions)>(
                  _device, particleCount),
          .velocities =
              createSharedBufferFromPointerType<decltype(sph::cuda::FluidParticlesDataImportedBuffer::velocities)>(
                  _device, particleCount),
          .densities =
              createSharedBufferFromPointerType<decltype(sph::cuda::FluidParticlesDataImportedBuffer::densities)>(
                  _device, particleCount),
          .radii = createSharedBufferFromPointerType<decltype(sph::cuda::FluidParticlesDataImportedBuffer::radii)>(
              _device, particleCount)}
{
}

FluidParticleRenderSystem::~FluidParticleRenderSystem() noexcept
{
    _device.logicalDevice.destroyPipelineLayout(_pipelineLayout);
}

auto FluidParticleRenderSystem::getImportedMemory() const -> sph::cuda::FluidParticlesDataImportedBuffer
{
    return {
        .positions = _particleBuffer.positions.getImportedMemory(),
        .velocities = _particleBuffer.velocities.getImportedMemory(),
        .densities = _particleBuffer.densities.getImportedMemory(),
        .radii = _particleBuffer.radii.getImportedMemory(),
    };
}

auto FluidParticleRenderSystem::createPipeline(const Device& device,
                                               vk::RenderPass renderPass,
                                               vk::PipelineLayout pipelineLayout) -> std::unique_ptr<Pipeline>
{
    static constexpr auto inputAssemblyInfo =
        vk::PipelineInputAssemblyStateCreateInfo {.topology = vk::PrimitiveTopology::eTriangleList,
                                                  .primitiveRestartEnable = vk::False};

    static constexpr auto viewportInfo = vk::PipelineViewportStateCreateInfo {.viewportCount = 1, .scissorCount = 1};
    static constexpr auto rasterizationInfo =
        vk::PipelineRasterizationStateCreateInfo {.depthClampEnable = vk::False,
                                                  .rasterizerDiscardEnable = vk::False,
                                                  .polygonMode = vk::PolygonMode::eFill,
                                                  .cullMode = vk::CullModeFlagBits::eBack,
                                                  .frontFace = vk::FrontFace::eCounterClockwise,
                                                  .depthBiasEnable = vk::False,
                                                  .lineWidth = 1.F};

    static constexpr auto multisamplingInfo =
        vk::PipelineMultisampleStateCreateInfo {.rasterizationSamples = vk::SampleCountFlagBits::e1,
                                                .sampleShadingEnable = vk::False};
    static constexpr auto colorBlendAttachment = vk::PipelineColorBlendAttachmentState {
        .blendEnable = vk::False,
        .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
        .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
        .dstAlphaBlendFactor = vk::BlendFactor::eZero,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};

    static constexpr auto colorBlendInfo =
        vk::PipelineColorBlendStateCreateInfo {.logicOpEnable = vk::False,
                                               .logicOp = vk::LogicOp::eCopy,
                                               .attachmentCount = 1,
                                               .pAttachments = &colorBlendAttachment};

    static constexpr auto depthStencilInfo =
        vk::PipelineDepthStencilStateCreateInfo {.depthTestEnable = vk::True,
                                                 .depthWriteEnable = vk::True,
                                                 .depthCompareOp = vk::CompareOp::eLess,
                                                 .depthBoundsTestEnable = vk::False,
                                                 .stencilTestEnable = vk::False};

    return std::make_unique<Pipeline>(
        device,
        PipelineConfig {.vertexShaderPath = config::shaderPath / "FluidParticle.vert.spv",
                        .fragmentShaderPath = config::shaderPath / "FluidParticle.frag.spv",
                        .vertexBindingDescriptions = {},
                        .vertexAttributeDescriptions = {},
                        .inputAssemblyInfo = inputAssemblyInfo,
                        .viewportInfo = viewportInfo,
                        .rasterizationInfo = rasterizationInfo,
                        .multisamplingInfo = multisamplingInfo,
                        .colorBlendInfo = colorBlendInfo,
                        .depthStencilInfo = depthStencilInfo,
                        .pipelineLayout = pipelineLayout,
                        .renderPass = renderPass,
                        .subpass = 0});
}

auto FluidParticleRenderSystem::createPipelineLayout(const Device& device, vk::DescriptorSetLayout setLayout)
    -> vk::PipelineLayout
{
    const auto pipelineLayoutInfo = vk::PipelineLayoutCreateInfo {.setLayoutCount = 1, .pSetLayouts = &setLayout};
    return expect(device.logicalDevice.createPipelineLayout(pipelineLayoutInfo),
                  vk::Result::eSuccess,
                  "Can't create pipeline layout");
}

auto FluidParticleRenderSystem::render(const FrameInfo& frameInfo) const -> void
{
    frameInfo.commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipeline->getHandle());

    DescriptorWriter(*_descriptorLayout)
        .writeBuffer(0, frameInfo.vertUbo.getDescriptorInfo())
        .writeBuffer(1, frameInfo.fragUbo.getDescriptorInfo())
        .writeBuffer(2, _particleBuffer.positions.getDescriptorInfo())
        .writeBuffer(3, _particleBuffer.velocities.getDescriptorInfo())
        .writeBuffer(4, _particleBuffer.radii.getDescriptorInfo())
        .push(frameInfo.commandBuffer, _pipelineLayout);

    frameInfo.commandBuffer.draw(6, frameInfo.scene.getParticleCount(), 0, 0);
}
}
