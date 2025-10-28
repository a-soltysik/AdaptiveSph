// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include "RenderSystem.h"

#include <filesystem>
#include <glm/ext/vector_float3.hpp>
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
#include "panda/gfx/Vertex.h"
#include "panda/gfx/object/Mesh.h"
#include "panda/gfx/object/Object.h"
#include "panda/gfx/object/Surface.h"
#include "panda/gfx/object/Texture.h"
#include "panda/internal/config.h"
#include "panda/utils/Utils.h"
#include "panda/utils/format/gfx/api/ResultFormatter.h"  // NOLINT(misc-include-cleaner)

namespace panda::gfx
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
    static constexpr auto inputAssemblyInfo =
        vk::PipelineInputAssemblyStateCreateInfo {.topology = vk::PrimitiveTopology::eTriangleList,
                                                  .primitiveRestartEnable = vk::False};

    static constexpr auto viewportInfo = vk::PipelineViewportStateCreateInfo {
        .viewportCount = 1,
        .scissorCount = 1,
    };
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
    static constexpr auto pushConstantData = vk::PushConstantRange {.stageFlags = vk::ShaderStageFlagBits::eVertex,
                                                                    .offset = 0,
                                                                    .size = sizeof(PushConstantData)};

    const auto pipelineLayoutInfo = vk::PipelineLayoutCreateInfo {.setLayoutCount = 1,
                                                                  .pSetLayouts = &setLayout,
                                                                  .pushConstantRangeCount = 1,
                                                                  .pPushConstantRanges = &pushConstantData};
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
