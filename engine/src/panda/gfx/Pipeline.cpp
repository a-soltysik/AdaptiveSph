// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include "panda/gfx/Pipeline.h"

#include <array>
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "Shader.h"
#include "panda/Logger.h"
#include "panda/gfx/Device.h"
#include "panda/utils/format/gfx/api/ResultFormatter.h"  // NOLINT(misc-include-cleaner)

namespace panda::gfx
{

Pipeline::Pipeline(const Device& device, const PipelineConfig& config)
    : _pipeline {createPipeline(device, config)},
      _device {device}
{
}

Pipeline::~Pipeline() noexcept
{
    log::Info("Destroying pipeline");
    _device.logicalDevice.destroy(_pipeline);
}

auto Pipeline::createPipeline(const Device& device, const PipelineConfig& config) -> vk::Pipeline
{
    const auto vertexShader = Shader::createFromFile(device.logicalDevice, config.vertexShaderPath);
    const auto fragmentShader = Shader::createFromFile(device.logicalDevice, config.fragmentShaderPath);

    auto shaderStages = std::vector<vk::PipelineShaderStageCreateInfo> {};

    if (vertexShader.has_value())
    {
        shaderStages.emplace_back(vk::PipelineShaderStageCreateInfo {.stage = vk::ShaderStageFlagBits::eVertex,
                                                                     .module = vertexShader->module,
                                                                     .pName = Shader::getEntryPointName()});
    }
    if (fragmentShader.has_value())
    {
        shaderStages.emplace_back(vk::PipelineShaderStageCreateInfo {.stage = vk::ShaderStageFlagBits::eFragment,
                                                                     .module = fragmentShader->module,
                                                                     .pName = Shader::getEntryPointName()});
    }

    static constexpr auto dynamicStates = std::array {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    static constexpr auto dynamicState = vk::PipelineDynamicStateCreateInfo {.dynamicStateCount = dynamicStates.size(),
                                                                             .pDynamicStates = dynamicStates.data()};

    const auto vertexInputInfo = vk::PipelineVertexInputStateCreateInfo {
        .vertexBindingDescriptionCount = static_cast<uint32_t>(config.vertexBindingDescriptions.size()),
        .pVertexBindingDescriptions = config.vertexBindingDescriptions.data(),
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(config.vertexAttributeDescriptions.size()),
        .pVertexAttributeDescriptions = config.vertexAttributeDescriptions.data()};

    const auto pipelineInfo = vk::GraphicsPipelineCreateInfo {.stageCount = static_cast<uint32_t>(shaderStages.size()),
                                                              .pStages = shaderStages.data(),
                                                              .pVertexInputState = &vertexInputInfo,
                                                              .pInputAssemblyState = &config.inputAssemblyInfo,
                                                              .pViewportState = &config.viewportInfo,
                                                              .pRasterizationState = &config.rasterizationInfo,
                                                              .pMultisampleState = &config.multisamplingInfo,
                                                              .pDepthStencilState = &config.depthStencilInfo,
                                                              .pColorBlendState = &config.colorBlendInfo,
                                                              .pDynamicState = &dynamicState,
                                                              .layout = config.pipelineLayout,
                                                              .renderPass = config.renderPass,
                                                              .subpass = config.subpass};

    return expect(device.logicalDevice.createGraphicsPipeline(nullptr, pipelineInfo),
                  vk::Result::eSuccess,
                  "Cannot create pipeline");
}

auto Pipeline::getHandle() const noexcept -> const vk::Pipeline&
{
    return _pipeline;
}

}
