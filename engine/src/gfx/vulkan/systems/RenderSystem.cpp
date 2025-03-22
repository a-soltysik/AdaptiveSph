// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include "panda/gfx/vulkan/systems/RenderSystem.h"

#include <filesystem>
#include <glm/ext/vector_float3.hpp>
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
#include "panda/gfx/vulkan/Vertex.h"
#include "panda/gfx/vulkan/object/Mesh.h"
#include "panda/gfx/vulkan/object/Object.h"
#include "panda/gfx/vulkan/object/Surface.h"
#include "panda/gfx/vulkan/object/Texture.h"
#include "panda/internal/config.h"
#include "panda/utils/Utils.h"
#include "panda/utils/format/gfx/api/vulkan/ResultFormatter.h"  // NOLINT(misc-include-cleaner)

namespace panda::gfx::vulkan
{

namespace
{

struct PushConstantData
{
    alignas(16) glm::vec3 translation;
    alignas(16) glm::vec3 scale;
    alignas(16) glm::vec3 rotation;
};

}

RenderSystem::RenderSystem(const Device& device, vk::RenderPass renderPass)
    : _device {device},
      _descriptorLayout {
          DescriptorSetLayout::Builder(_device)
              .addBinding(0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex)
              .addBinding(1, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eFragment)
              .addBinding(2, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment)
              .build(vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR)},
      _pipelineLayout {createPipelineLayout(_device, _descriptorLayout->getDescriptorSetLayout())},
      _pipeline {createPipeline(_device, renderPass, _pipelineLayout)}

{
}

RenderSystem::~RenderSystem() noexcept
{
    _device.logicalDevice.destroyPipelineLayout(_pipelineLayout);
}

auto RenderSystem::createPipeline(const Device& device, vk::RenderPass renderPass, vk::PipelineLayout pipelineLayout)
    -> std::unique_ptr<Pipeline>
{
    const auto inputAssemblyInfo =
        vk::PipelineInputAssemblyStateCreateInfo {{}, vk::PrimitiveTopology::eTriangleList, vk::False};

    const auto viewportInfo = vk::PipelineViewportStateCreateInfo {{}, 1, {}, 1, {}};
    const auto rasterizationInfo = vk::PipelineRasterizationStateCreateInfo {{},
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

    const auto multisamplingInfo = vk::PipelineMultisampleStateCreateInfo {{}, vk::SampleCountFlagBits::e1, vk::False};
    const auto colorBlendAttachment =
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

    const auto depthStencilInfo =
        vk::PipelineDepthStencilStateCreateInfo {{}, vk::True, vk::True, vk::CompareOp::eLess, vk::False, vk::False};

    return std::make_unique<Pipeline>(
        device,
        PipelineConfig {.vertexShaderPath = config::shaderPath / "basic.vert.spv",
                        .fragmentShaderPath = config::shaderPath / "basic.frag.spv",
                        .vertexBindingDescriptions = {Vertex::getBindingDescription()},
                        .vertexAttributeDescriptions = utils::fromArray(Vertex::getAttributeDescriptions()),
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

auto RenderSystem::createPipelineLayout(const Device& device, vk::DescriptorSetLayout setLayout) -> vk::PipelineLayout
{
    const auto pushConstantData = vk::PushConstantRange {vk::ShaderStageFlagBits::eVertex, 0, sizeof(PushConstantData)};

    const auto pipelineLayoutInfo = vk::PipelineLayoutCreateInfo {{}, setLayout, pushConstantData};
    return expect(device.logicalDevice.createPipelineLayout(pipelineLayoutInfo),
                  vk::Result::eSuccess,
                  "Can't create pipeline layout");
}

auto RenderSystem::render(const FrameInfo& frameInfo) const -> void
{
    frameInfo.commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipeline->getHandle());

    for (const auto& object : frameInfo.scene.getObjects())
    {
        renderObject(*object, frameInfo);
    }
    renderObject(frameInfo.scene.getDomain(), frameInfo);
}

auto RenderSystem::renderObject(const Object& object, const FrameInfo& frameInfo) const -> void
{
    const auto push = PushConstantData {.translation = object.transform.translation,
                                        .scale = object.transform.scale,
                                        .rotation = object.transform.rotation};

    frameInfo.commandBuffer.pushConstants<PushConstantData>(_pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, push);

    for (const auto& surface : object.getSurfaces())
    {
        DescriptorWriter(*_descriptorLayout)
            .writeBuffer(0, frameInfo.vertUbo.getDescriptorInfo())
            .writeBuffer(1, frameInfo.fragUbo.getDescriptorInfo())
            .writeImage(2, surface.getTexture().getDescriptorImageInfo())
            .push(frameInfo.commandBuffer, _pipelineLayout);

        surface.getMesh().bind(frameInfo.commandBuffer);
        surface.getMesh().draw(frameInfo.commandBuffer);
    }
}

}
