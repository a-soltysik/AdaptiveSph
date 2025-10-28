#pragma once

// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include <cstdint>
#include <filesystem>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "Device.h"
#include "panda/Common.h"

namespace panda::gfx
{

struct PipelineConfig
{
    std::filesystem::path vertexShaderPath;
    std::filesystem::path fragmentShaderPath;

    std::vector<vk::VertexInputBindingDescription> vertexBindingDescriptions;
    std::vector<vk::VertexInputAttributeDescription> vertexAttributeDescriptions;
    vk::PipelineInputAssemblyStateCreateInfo inputAssemblyInfo;
    vk::PipelineViewportStateCreateInfo viewportInfo;
    vk::PipelineRasterizationStateCreateInfo rasterizationInfo;
    vk::PipelineMultisampleStateCreateInfo multisamplingInfo;
    vk::PipelineColorBlendStateCreateInfo colorBlendInfo;
    vk::PipelineDepthStencilStateCreateInfo depthStencilInfo;
    vk::PipelineLayout pipelineLayout;
    vk::RenderPass renderPass;
    uint32_t subpass = 0;
};

class Pipeline
{
public:
    Pipeline(const Device& device, const PipelineConfig& config);
    PD_DELETE_ALL(Pipeline);
    ~Pipeline() noexcept;

    [[nodiscard]] auto getHandle() const noexcept -> const vk::Pipeline&;

private:
    [[nodiscard]] static auto createPipeline(const Device& device, const PipelineConfig& config) -> vk::Pipeline;

    vk::Pipeline _pipeline;
    const Device& _device;
};

}
