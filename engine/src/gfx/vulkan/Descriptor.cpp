// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include "panda/gfx/vulkan/Descriptor.h"

#include <fmt/format.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "panda/gfx/vulkan/Device.h"
#include "panda/utils/format/gfx/api/vulkan/ResultFormatter.h"  // NOLINT(misc-include-cleaner)

namespace panda::gfx::vulkan
{

DescriptorSetLayout::Builder::Builder(const Device& device)
    : _device {device}
{
}

auto DescriptorSetLayout::Builder::addBinding(uint32_t binding,
                                              vk::DescriptorType descriptorType,
                                              vk::ShaderStageFlags stageFlags,
                                              uint32_t count) -> Builder&
{
    expect(!_bindings.contains(binding), fmt::format("Binding: {} already in use", binding));
    const auto layoutBinding = vk::DescriptorSetLayoutBinding {binding, descriptorType, count, stageFlags};
    _bindings.insert({binding, layoutBinding});

    return *this;
}

auto DescriptorSetLayout::Builder::build(vk::DescriptorSetLayoutCreateFlags flags) const
    -> std::unique_ptr<DescriptorSetLayout>
{
    return std::make_unique<DescriptorSetLayout>(_device, _bindings, flags);
}

DescriptorSetLayout::DescriptorSetLayout(const Device& device,
                                         const std::unordered_map<uint32_t, vk::DescriptorSetLayoutBinding>& bindings,
                                         vk::DescriptorSetLayoutCreateFlags flags)
    : _bindings {bindings},
      _device {device},
      _descriptorSetLayout {createDescriptorSetLayout(device, bindings, flags)}
{
}

DescriptorSetLayout::~DescriptorSetLayout() noexcept
{
    _device.logicalDevice.destroyDescriptorSetLayout(_descriptorSetLayout);
}

auto DescriptorSetLayout::createDescriptorSetLayout(
    const Device& device,
    const std::unordered_map<uint32_t, vk::DescriptorSetLayoutBinding>& bindings,
    vk::DescriptorSetLayoutCreateFlags flags) -> vk::DescriptorSetLayout
{
    auto layoutBindings = std::vector<vk::DescriptorSetLayoutBinding> {};
    layoutBindings.reserve(bindings.size());

    std::ranges::transform(bindings, std::back_inserter(layoutBindings), [](const auto& binding) {
        return binding.second;
    });

    const auto descriptorSetLayoutInfo = vk::DescriptorSetLayoutCreateInfo {flags, layoutBindings};
    return expect(device.logicalDevice.createDescriptorSetLayout(descriptorSetLayoutInfo),
                  vk::Result::eSuccess,
                  "Failed to create descriptor set layout");
}

auto DescriptorSetLayout::getDescriptorSetLayoutBinding(uint32_t binding) const -> const vk::DescriptorSetLayoutBinding&
{
    return expectNot(_bindings.find(binding), _bindings.cend(), fmt::format("Binding: {} does not exist", binding))
        ->second;
}

auto DescriptorSetLayout::getDescriptorSetLayout() const noexcept -> vk::DescriptorSetLayout
{
    return _descriptorSetLayout;
}

DescriptorPool::Builder::Builder(const Device& device)
    : _device {device}
{
}

auto DescriptorPool::Builder::addPoolSize(vk::DescriptorType descriptorType, uint32_t count) -> Builder&
{
    _poolSizes.emplace_back(descriptorType, count);
    return *this;
}

auto DescriptorPool::Builder::build(uint32_t maxSets, vk::DescriptorPoolCreateFlags flags)
    -> std::unique_ptr<DescriptorPool>
{
    return std::make_unique<DescriptorPool>(_device, flags, maxSets, _poolSizes);
}

DescriptorPool::DescriptorPool(const Device& device,
                               vk::DescriptorPoolCreateFlags poolFlags,
                               uint32_t maxSets,
                               const std::vector<vk::DescriptorPoolSize>& poolSizes)
    : _device {device},
      _descriptorPool {createDescriptorPool(device, maxSets, poolFlags, poolSizes)}
{
}

DescriptorPool::~DescriptorPool() noexcept
{
    _device.logicalDevice.destroyDescriptorPool(_descriptorPool);
}

auto DescriptorPool::allocateDescriptor(vk::DescriptorSetLayout descriptorSetLayout,
                                        vk::DescriptorSet& descriptor) const -> bool
{
    const auto allocInfo = vk::DescriptorSetAllocateInfo {_descriptorPool, descriptorSetLayout};

    return shouldBe(_device.logicalDevice.allocateDescriptorSets(&allocInfo, &descriptor),
                    vk::Result::eSuccess,
                    "Failed to allocate descriptor sets");
}

auto DescriptorPool::createDescriptorPool(const Device& device,
                                          uint32_t maxSets,
                                          vk::DescriptorPoolCreateFlags poolFlags,
                                          const std::vector<vk::DescriptorPoolSize>& poolSizes) -> vk::DescriptorPool
{
    const auto descriptorPoolInfo = vk::DescriptorPoolCreateInfo {poolFlags, maxSets, poolSizes};

    return expect(device.logicalDevice.createDescriptorPool(descriptorPoolInfo),
                  vk::Result::eSuccess,
                  "Failed to create descriptor pool!");
}

auto DescriptorPool::freeDescriptors(const std::vector<vk::DescriptorSet>& descriptors) const -> void
{
    _device.logicalDevice.freeDescriptorSets(_descriptorPool, descriptors);
}

auto DescriptorPool::resetPool() -> void
{
    _device.logicalDevice.resetDescriptorPool(_descriptorPool);
}

auto DescriptorPool::getHandle() const noexcept -> vk::DescriptorPool
{
    return _descriptorPool;
}

DescriptorWriter::DescriptorWriter(const DescriptorSetLayout& setLayout)
    : _setLayout {setLayout}
{
}

auto DescriptorWriter::writeBuffer(uint32_t binding, const vk::DescriptorBufferInfo& bufferInfo) -> DescriptorWriter&
{
    const auto& bindingDescription = _setLayout.getDescriptorSetLayoutBinding(binding);

    _writes.push_back(
        vk::WriteDescriptorSet {{}, binding, {}, 1, bindingDescription.descriptorType, {}, &bufferInfo, {}});
    return *this;
}

auto DescriptorWriter::writeImage(uint32_t binding, const vk::DescriptorImageInfo& imageInfo) -> DescriptorWriter&
{
    const auto& bindingDescription = _setLayout.getDescriptorSetLayoutBinding(binding);

    _writes.push_back(
        vk::WriteDescriptorSet {{}, binding, {}, 1, bindingDescription.descriptorType, &imageInfo, {}, {}});
    return *this;
}

auto DescriptorWriter::push(vk::CommandBuffer commandBuffer, vk::PipelineLayout layout) -> void
{
    commandBuffer.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, layout, 0, _writes);
}

}
