#pragma once

// clang-format off
#include "panda/gfx/vulkan/SharedBuffer.hpp"
#include "panda/utils/Assert.h"
// clang-format on

#include <panda/gfx/vulkan/object/Object.h>

#include <memory>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "panda/Common.h"
#include "panda/gfx/vulkan/Pipeline.h"

namespace panda::gfx::vulkan
{
class DescriptorSetLayout;
class Device;
struct FrameInfo;

class RenderSystem
{
public:
    RenderSystem(const Device& device, vk::RenderPass renderPass);
    PD_DELETE_ALL(RenderSystem);
    ~RenderSystem() noexcept;

    auto render(const FrameInfo& frameInfo) const -> void;

private:
    static auto createPipelineLayout(const Device& device, vk::DescriptorSetLayout setLayout) -> vk::PipelineLayout;
    static auto createPipeline(const Device& device, vk::RenderPass renderPass, vk::PipelineLayout pipelineLayout)
        -> std::unique_ptr<Pipeline>;

    auto renderObject(const Object& object, const FrameInfo& frameInfo) const -> void;

    const Device& _device;
    std::unique_ptr<DescriptorSetLayout> _descriptorLayout;
    vk::PipelineLayout _pipelineLayout;
    std::unique_ptr<Pipeline> _pipeline;
};

}
