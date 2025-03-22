// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include "panda/gfx/vulkan/Pipeline.h"

#include <array>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "panda/Logger.h"
#include "panda/gfx/vulkan/Device.h"
#include "panda/gfx/vulkan/Shader.h"
#include "panda/utils/format/gfx/api/vulkan/ResultFormatter.h"  // NOLINT(misc-include-cleaner)

namespace panda::gfx::vulkan
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
        shaderStages.emplace_back(vk::PipelineShaderStageCreateFlags {},
                                  vk::ShaderStageFlagBits::eVertex,
                                  vertexShader->module,
                                  Shader::getEntryPointName());
    }
    if (fragmentShader.has_value())
    {
        shaderStages.emplace_back(vk::PipelineShaderStageCreateFlags {},
                                  vk::ShaderStageFlagBits::eFragment,
                                  fragmentShader->module,
                                  Shader::getEntryPointName());
    }

    const auto dynamicStates = std::array {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    const auto dynamicState = vk::PipelineDynamicStateCreateInfo {{}, dynamicStates};

    const auto vertexInputInfo = vk::PipelineVertexInputStateCreateInfo {{},
                                                                         config.vertexBindingDescriptions,
                                                                         config.vertexAttributeDescriptions};

    const auto pipelineInfo = vk::GraphicsPipelineCreateInfo {{},
                                                              shaderStages,
                                                              &vertexInputInfo,
                                                              &config.inputAssemblyInfo,
                                                              {},
                                                              &config.viewportInfo,
                                                              &config.rasterizationInfo,
                                                              &config.multisamplingInfo,
                                                              &config.depthStencilInfo,
                                                              &config.colorBlendInfo,
                                                              &dynamicState,
                                                              config.pipelineLayout,
                                                              config.renderPass,
                                                              config.subpass};

    return expect(device.logicalDevice.createGraphicsPipeline(nullptr, pipelineInfo),
                  vk::Result::eSuccess,
                  "Cannot create pipeline");
}

auto Pipeline::getHandle() const noexcept -> const vk::Pipeline&
{
    return _pipeline;
}

}
