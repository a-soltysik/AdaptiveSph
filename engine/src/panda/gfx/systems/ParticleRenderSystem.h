#pragma once

// clang-format off
#include "panda/utils/Assert.h" // NOLINT(misc-include-cleaner)
// clang-format on

#include <cuda/Simulation.cuh>
#include <memory>
#include <type_traits>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "panda/Common.h"
#include "panda/gfx/Pipeline.h"
#include "panda/gfx/SharedBuffer.hpp"

namespace panda::gfx
{
class DescriptorSetLayout;
class Device;
struct FrameInfo;

class ParticleRenderSystem
{
public:
    ParticleRenderSystem(const Device& device, vk::RenderPass renderPass, size_t particleCount);
    PD_DELETE_ALL(ParticleRenderSystem);
    ~ParticleRenderSystem() noexcept;

    auto render(const FrameInfo& frameInfo) const -> void;
    [[nodiscard]] auto getImportedMemory() const -> sph::cuda::ParticlesDataBuffer;

private:
    static auto createPipelineLayout(const Device& device, vk::DescriptorSetLayout setLayout) -> vk::PipelineLayout;
    static auto createPipeline(const Device& device, vk::RenderPass renderPass, vk::PipelineLayout pipelineLayout)
        -> std::unique_ptr<Pipeline>;

    template <typename T>
    static auto createSharedBufferFromPointerType(const Device& device, size_t particleCount) -> SharedBuffer
    {
        return {device,
                sizeof(std::remove_pointer_t<T>),
                particleCount,
                vk::MemoryPropertyFlagBits::eDeviceLocal,
                device.physicalDevice.getProperties().limits.minUniformBufferOffsetAlignment};
    }

    struct ParticlesDataBuffer
    {
        SharedBuffer positions;
        SharedBuffer velocities;
        SharedBuffer accelerations;
        SharedBuffer densities;
        SharedBuffer radiuses;
        SharedBuffer smoothingRadiuses;
        SharedBuffer masses;
    };

    const Device& _device;
    std::unique_ptr<DescriptorSetLayout> _descriptorLayout;
    vk::PipelineLayout _pipelineLayout;
    std::unique_ptr<Pipeline> _pipeline;
    ParticlesDataBuffer _particleBuffer;
};

}
