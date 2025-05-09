// clang-format off
#include <cuda/Simulation.cuh>
#include "panda/utils/Assert.h"
// clang-format on

#include "panda/gfx/vulkan/systems/ParticleRenderSystem.h"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "panda/gfx/vulkan/Buffer.h"
#include "panda/gfx/vulkan/Descriptor.h"
#include "panda/gfx/vulkan/Device.h"
#include "panda/gfx/vulkan/FrameInfo.h"
#include "panda/gfx/vulkan/Pipeline.h"
#include "panda/gfx/vulkan/Scene.h"
#include "panda/internal/config.h"
#include "panda/utils/format/gfx/api/vulkan/ResultFormatter.h"  // NOLINT(misc-include-cleaner, unused-includes)

namespace panda::gfx::vulkan
{
ParticleRenderSystem::ParticleRenderSystem(const Device& device, vk::RenderPass renderPass, size_t particleCount)
    : _device {device},
      _descriptorLayout {DescriptorSetLayout::Builder(_device)
                             .addBinding(0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex)
                             .addBinding(1, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eFragment)
                             .addBinding(2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eVertex)
                             .addBinding(3, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eVertex)
                             .addBinding(4, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eVertex)
                             .addBinding(5, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eVertex)
                             .build(vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR)},
      _pipelineLayout {createPipelineLayout(_device, _descriptorLayout->getDescriptorSetLayout())},
      _pipeline {createPipeline(_device, renderPass, _pipelineLayout)},
      _particleBuffer {
          .positions =
              createSharedBufferFromPointerType<decltype(sph::cuda::ParticlesData::positions)>(_device, particleCount),
          .predictedPositions =
              createSharedBufferFromPointerType<decltype(sph::cuda::ParticlesData::predictedPositions)>(_device,
                                                                                                        particleCount),
          .velocities =
              createSharedBufferFromPointerType<decltype(sph::cuda::ParticlesData::velocities)>(_device, particleCount),
          .forces =
              createSharedBufferFromPointerType<decltype(sph::cuda::ParticlesData::forces)>(_device, particleCount),
          .densities =
              createSharedBufferFromPointerType<decltype(sph::cuda::ParticlesData::densities)>(_device, particleCount),
          .nearDensities = createSharedBufferFromPointerType<decltype(sph::cuda::ParticlesData::nearDensities)>(
              _device, particleCount),
          .pressures =
              createSharedBufferFromPointerType<decltype(sph::cuda::ParticlesData::pressures)>(_device, particleCount),
          .radiuses =
              createSharedBufferFromPointerType<decltype(sph::cuda::ParticlesData::radiuses)>(_device, particleCount),
          .smoothingRadiuses = createSharedBufferFromPointerType<decltype(sph::cuda::ParticlesData::smoothingRadiuses)>(
              _device, particleCount),
          .masses =
              createSharedBufferFromPointerType<decltype(sph::cuda::ParticlesData::masses)>(_device, particleCount),
          .refinementLevels = createSharedBufferFromPointerType<decltype(sph::cuda::ParticlesData::refinementLevels)>(
              _device, particleCount)}
{
}

ParticleRenderSystem::~ParticleRenderSystem() noexcept
{
    _device.logicalDevice.destroyPipelineLayout(_pipelineLayout);
}

auto ParticleRenderSystem::getImportedMemory() const -> sph::cuda::ParticlesDataBuffer
{
    return {
        .positions = _particleBuffer.positions.getImportedMemory(),
        .predictedPositions = _particleBuffer.predictedPositions.getImportedMemory(),
        .velocities = _particleBuffer.velocities.getImportedMemory(),
        .forces = _particleBuffer.forces.getImportedMemory(),
        .densities = _particleBuffer.densities.getImportedMemory(),
        .nearDensities = _particleBuffer.nearDensities.getImportedMemory(),
        .pressures = _particleBuffer.pressures.getImportedMemory(),
        .radiuses = _particleBuffer.radiuses.getImportedMemory(),
        .smoothingRadiuses = _particleBuffer.smoothingRadiuses.getImportedMemory(),
        .masses = _particleBuffer.masses.getImportedMemory(),
        .refinementLevels = _particleBuffer.refinementLevels.getImportedMemory(),
    };
}

auto ParticleRenderSystem::createPipeline(const Device& device,
                                          vk::RenderPass renderPass,
                                          vk::PipelineLayout pipelineLayout) -> std::unique_ptr<Pipeline>
{
    static constexpr auto inputAssemblyInfo =
        vk::PipelineInputAssemblyStateCreateInfo {{}, vk::PrimitiveTopology::eTriangleList, vk::False};

    static constexpr auto viewportInfo = vk::PipelineViewportStateCreateInfo {{}, 1, {}, 1, {}};
    static constexpr auto rasterizationInfo =
        vk::PipelineRasterizationStateCreateInfo {{},
                                                  vk::False,
                                                  vk::False,
                                                  vk::PolygonMode::eFill,
                                                  vk::CullModeFlagBits::eBack,
                                                  vk::FrontFace::eCounterClockwise,
                                                  vk::False,
                                                  {},
                                                  {},
                                                  {},
                                                  1.F};

    static constexpr auto multisamplingInfo =
        vk::PipelineMultisampleStateCreateInfo {{}, vk::SampleCountFlagBits::e1, vk::False};
    static constexpr auto colorBlendAttachment =
        vk::PipelineColorBlendAttachmentState {vk::False,
                                               vk::BlendFactor::eSrcAlpha,
                                               vk::BlendFactor::eOneMinusSrcAlpha,
                                               vk::BlendOp::eAdd,
                                               vk::BlendFactor::eOne,
                                               vk::BlendFactor::eZero,
                                               vk::BlendOp::eAdd,
                                               vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                                   vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};

    const auto colorBlendInfo =
        vk::PipelineColorBlendStateCreateInfo {{}, vk::False, vk::LogicOp::eCopy, colorBlendAttachment};

    static constexpr auto depthStencilInfo =
        vk::PipelineDepthStencilStateCreateInfo {{}, vk::True, vk::True, vk::CompareOp::eLess, vk::False, vk::False};

    return std::make_unique<Pipeline>(device,
                                      PipelineConfig {.vertexShaderPath = config::shaderPath / "particle.vert.spv",
                                                      .fragmentShaderPath = config::shaderPath / "particle.frag.spv",
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

auto ParticleRenderSystem::createPipelineLayout(const Device& device, vk::DescriptorSetLayout setLayout)
    -> vk::PipelineLayout
{
    const auto pipelineLayoutInfo = vk::PipelineLayoutCreateInfo {{}, setLayout};
    return expect(device.logicalDevice.createPipelineLayout(pipelineLayoutInfo),
                  vk::Result::eSuccess,
                  "Can't create pipeline layout");
}

auto ParticleRenderSystem::render(const FrameInfo& frameInfo) const -> void
{
    frameInfo.commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipeline->getHandle());

    DescriptorWriter(*_descriptorLayout)
        .writeBuffer(0, frameInfo.vertUbo.getDescriptorInfo())
        .writeBuffer(1, frameInfo.fragUbo.getDescriptorInfo())
        .writeBuffer(2, _particleBuffer.positions.getDescriptorInfo())
        .writeBuffer(3, _particleBuffer.velocities.getDescriptorInfo())
        .writeBuffer(4, _particleBuffer.radiuses.getDescriptorInfo())
        .writeBuffer(5, _particleBuffer.forces.getDescriptorInfo())
        .push(frameInfo.commandBuffer, _pipelineLayout);

    frameInfo.commandBuffer.draw(6, frameInfo.scene.getParticleCount(), 0, 0);
}
}
