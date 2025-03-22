#pragma once

// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "Device.h"
#include "panda/Common.h"

namespace panda::gfx::vulkan
{

class DescriptorSetLayout
{
public:
    class Builder
    {
    public:
        explicit Builder(const Device& device);
        [[nodiscard]] auto addBinding(uint32_t binding,
                                      vk::DescriptorType descriptorType,
                                      vk::ShaderStageFlags stageFlags,
                                      uint32_t count = 1) -> Builder&;
        [[nodiscard]] auto build(vk::DescriptorSetLayoutCreateFlags flags = {}) const
            -> std::unique_ptr<DescriptorSetLayout>;

    private:
        const Device& _device;
        std::unordered_map<uint32_t, vk::DescriptorSetLayoutBinding> _bindings;
    };

    DescriptorSetLayout(const Device& device,
                        const std::unordered_map<uint32_t, vk::DescriptorSetLayoutBinding>& bindings,
                        vk::DescriptorSetLayoutCreateFlags flags = {});

    PD_DELETE_ALL(DescriptorSetLayout);
    ~DescriptorSetLayout() noexcept;

    [[nodiscard]] auto getDescriptorSetLayout() const noexcept -> vk::DescriptorSetLayout;
    [[nodiscard]] auto getDescriptorSetLayoutBinding(uint32_t binding) const -> const vk::DescriptorSetLayoutBinding&;

private:
    [[nodiscard]] static auto createDescriptorSetLayout(
        const Device& device,
        const std::unordered_map<uint32_t, vk::DescriptorSetLayoutBinding>& bindings,
        vk::DescriptorSetLayoutCreateFlags flags) -> vk::DescriptorSetLayout;

    std::unordered_map<uint32_t, vk::DescriptorSetLayoutBinding> _bindings;
    const Device& _device;
    vk::DescriptorSetLayout _descriptorSetLayout;
};

class DescriptorPool
{
public:
    class Builder
    {
    public:
        explicit Builder(const Device& device);
        [[nodiscard]] auto addPoolSize(vk::DescriptorType descriptorType, uint32_t count) -> Builder&;
        [[nodiscard]] auto build(uint32_t maxSets, vk::DescriptorPoolCreateFlags flags = {})
            -> std::unique_ptr<DescriptorPool>;

    private:
        const Device& _device;
        std::vector<vk::DescriptorPoolSize> _poolSizes;
    };

    DescriptorPool(const Device& device,
                   vk::DescriptorPoolCreateFlags poolFlags,
                   uint32_t maxSets,
                   const std::vector<vk::DescriptorPoolSize>& poolSizes);

    PD_DELETE_ALL(DescriptorPool);
    ~DescriptorPool() noexcept;

    auto allocateDescriptor(vk::DescriptorSetLayout descriptorSetLayout, vk::DescriptorSet& descriptor) const -> bool;
    auto freeDescriptors(const std::vector<vk::DescriptorSet>& descriptors) const -> void;
    auto resetPool() -> void;
    [[nodiscard]] auto getHandle() const noexcept -> vk::DescriptorPool;

private:
    [[nodiscard]] static auto createDescriptorPool(const Device& device,
                                                   uint32_t maxSets,
                                                   vk::DescriptorPoolCreateFlags poolFlags,
                                                   const std::vector<vk::DescriptorPoolSize>& poolSizes)
        -> vk::DescriptorPool;
    const Device& _device;
    const vk::DescriptorPool _descriptorPool;
};

class DescriptorWriter
{
public:
    explicit DescriptorWriter(const DescriptorSetLayout& setLayout);

    [[nodiscard]] auto writeBuffer(uint32_t binding, const vk::DescriptorBufferInfo& bufferInfo) -> DescriptorWriter&;
    [[nodiscard]] auto writeImage(uint32_t binding, const vk::DescriptorImageInfo& imageInfo) -> DescriptorWriter&;

    auto push(vk::CommandBuffer commandBuffer, vk::PipelineLayout layout) -> void;

private:
    const DescriptorSetLayout& _setLayout;

    std::vector<vk::WriteDescriptorSet> _writes;
};

}
