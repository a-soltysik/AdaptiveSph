#pragma once

// clang-format off
#include "panda/utils/Assert.h" // NOLINT(misc-include-cleaner)
// clang-format on

#include <cstddef>
#include <glm/ext/vector_float3.hpp>
#include <memory>
#include <panda/gfx/vulkan/SharedBuffer.hpp>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "panda/Common.h"
#include "panda/gfx/vulkan/Pipeline.h"

namespace panda::gfx::vulkan
{
class DescriptorSetLayout;
class Device;
struct FrameInfo;

class ParticleRenderSystem
{
public:
    ParticleRenderSystem(const Device& device,
                         vk::RenderPass renderPass,
                         vk::DeviceSize particleSize,
                         size_t particleCount);
    PD_DELETE_ALL(ParticleRenderSystem);
    ~ParticleRenderSystem() noexcept;

    auto render(const FrameInfo& frameInfo) const -> void;
    [[nodiscard]] auto getImportedMemory() const -> const sph::cuda::ImportedMemory&;

private:
    static auto createPipelineLayout(const Device& device, vk::DescriptorSetLayout setLayout) -> vk::PipelineLayout;
    static auto createPipeline(const Device& device, vk::RenderPass renderPass, vk::PipelineLayout pipelineLayout)
        -> std::unique_ptr<Pipeline>;

    const Device& _device;
    std::unique_ptr<DescriptorSetLayout> _descriptorLayout;
    vk::PipelineLayout _pipelineLayout;
    std::unique_ptr<Pipeline> _pipeline;
    SharedBuffer _particleBuffer;
};

}
