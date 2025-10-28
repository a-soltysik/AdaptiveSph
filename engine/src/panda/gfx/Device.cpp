// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include "panda/gfx/Device.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <ranges>
#include <span>
#include <string_view>
#include <unordered_set>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "panda/Logger.h"
#include "panda/utils/format/gfx/api/ResultFormatter.h"  // NOLINT(misc-include-cleaner)

namespace panda::gfx
{

Device::Device(const vk::Instance& instance,
               const vk::SurfaceKHR& surface,
               std::span<const char* const> requiredExtensions)
    : physicalDevice {pickPhysicalDevice(instance, surface, requiredExtensions)},
      queueFamilies {expect(findQueueFamilies(physicalDevice, surface), "Queue families need to exist")},
      logicalDevice {createLogicalDevice(physicalDevice, queueFamilies, requiredExtensions)},
      graphicsQueue {logicalDevice.getQueue(queueFamilies.graphicsFamily, 0)},
      presentationQueue {logicalDevice.getQueue(queueFamilies.presentationFamily, 0)},
      commandPool {expect(logicalDevice.createCommandPool({.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                                           .queueFamilyIndex = queueFamilies.graphicsFamily}),
                          vk::Result::eSuccess,
                          "Can't create command pool")},
      _surface {surface}
{
}

auto Device::pickPhysicalDevice(const vk::Instance& instance,
                                const vk::SurfaceKHR& surface,
                                std::span<const char* const> requiredExtensions) -> vk::PhysicalDevice
{
    const auto devices =
        expect(instance.enumeratePhysicalDevices(), vk::Result::eSuccess, "Can't enumerate physical devices");

    const auto it = std::ranges::find_if(devices, [&surface, requiredExtensions](const auto& currentDevice) {
        return isDeviceSuitable(currentDevice, surface, requiredExtensions);
    });

    expectNot(it, devices.cend(), "None_ of physical devices is suitable");
    return *it;
}

auto Device::isDeviceSuitable(vk::PhysicalDevice device,
                              vk::SurfaceKHR surface,
                              std::span<const char* const> requiredExtensions) -> bool
{
    const auto queueFamilies = findQueueFamilies(device, surface);
    const auto swapChainSupport = querySwapChainSupport(device, surface);

    return queueFamilies && checkDeviceExtensionSupport(device, requiredExtensions) &&
           !swapChainSupport.formats.empty() && !swapChainSupport.presentationModes.empty() &&
           device.getFeatures().samplerAnisotropy > 0;
}

auto Device::findQueueFamilies(vk::PhysicalDevice device, vk::SurfaceKHR surface) -> std::optional<QueueFamilies>
{
    const auto queueFamilies = device.getQueueFamilyProperties();
    auto queueFamilyIndices = QueueFamilies {};

    auto isGraphicsSet = false;
    auto isPresentSet = false;

    for (auto i = uint32_t {}; i < static_cast<uint32_t>(queueFamilies.size()); i++)
    {
        if (queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics)
        {
            log::Info("Graphics queue index: {}", i);
            queueFamilyIndices.graphicsFamily = i;
            isGraphicsSet = true;
        }
        if (device.getSurfaceSupportKHR(i, surface).value != 0)
        {
            log::Info("Presentation queue index: {}", i);
            queueFamilyIndices.presentationFamily = i;
            isPresentSet = true;
        }
        if (isGraphicsSet && isPresentSet)
        {
            return queueFamilyIndices;
        }
    }

    return {};
}

auto Device::querySwapChainSupport(vk::PhysicalDevice device, vk::SurfaceKHR surface) -> SwapChainSupportDetails
{
    return {.capabilities = device.getSurfaceCapabilitiesKHR(surface).value,
            .formats = device.getSurfaceFormatsKHR(surface).value,
            .presentationModes = device.getSurfacePresentModesKHR(surface).value};
}

auto Device::checkDeviceExtensionSupport(vk::PhysicalDevice device, std::span<const char* const> requiredExtensions)
    -> bool
{
    const auto availableExtensions = device.enumerateDeviceExtensionProperties();
    if (availableExtensions.result != vk::Result::eSuccess)
    {
        log::Error("Can't enumerate device extensions: {}", availableExtensions.result);
        return false;
    }

    for (const auto* extension : requiredExtensions)
    {
        const auto it = std::ranges::find_if(
            availableExtensions.value,
            [extension](const auto& availableExtension) {
                return std::string_view {extension} == std::string_view {availableExtension};
            },
            &vk::ExtensionProperties::extensionName);

        if (it == availableExtensions.value.cend())
        {
            log::Warning("{} extension is unavailable", extension);
            return false;
        }
    }
    return true;
}

auto Device::createLogicalDevice(vk::PhysicalDevice device,
                                 const QueueFamilies& queueFamilies,
                                 std::span<const char* const> requiredExtensions) -> vk::Device
{
    static constexpr auto queuePriority = 1.F;

    const auto queueCreateInfos = std::ranges::to<std::vector<vk::DeviceQueueCreateInfo>>(
        queueFamilies.getUniqueQueueFamilies() | std::ranges::views::transform([](const auto queueFamily) {
            return vk::DeviceQueueCreateInfo {.queueFamilyIndex = queueFamily,
                                              .queueCount = 1,
                                              .pQueuePriorities = &queuePriority};
        }));

    static constexpr auto physicalDeviceFeatures = vk::PhysicalDeviceFeatures {.samplerAnisotropy = vk::True};

#ifdef __clang__
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wmissing-field-initializers"
#elifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
    //NOLINTBEGIN(clang-diagnostic-missing-designated-field-initializers)
    const auto createInfo =
        vk::DeviceCreateInfo {.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
                              .pQueueCreateInfos = queueCreateInfos.data(),
                              .enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
                              .ppEnabledExtensionNames = requiredExtensions.data(),
                              .pEnabledFeatures = &physicalDeviceFeatures};
//NOLINTEND(clang-diagnostic-missing-designated-field-initializers)
#ifdef __clang__
#    pragma clang diagnostic pop
#elifdef __GNUC__
#    pragma GCC diagnostic pop
#endif

    return expect(device.createDevice(createInfo), vk::Result::eSuccess, "Can't create physical device");
}

auto Device::querySwapChainSupport() const -> SwapChainSupportDetails
{
    return querySwapChainSupport(physicalDevice, _surface);
}

Device::~Device() noexcept
{
    log::Info("Destroying device");
    logicalDevice.destroy(commandPool);
    logicalDevice.destroy();
}

auto Device::findSupportedFormat(std::span<const vk::Format> candidates,
                                 vk::ImageTiling tiling,
                                 vk::FormatFeatureFlags features) const noexcept -> std::optional<vk::Format>
{
    for (const auto format : candidates)
    {
        const auto properties = physicalDevice.getFormatProperties(format);

        if (tiling == vk::ImageTiling::eLinear && (properties.linearTilingFeatures & features) == features)
        {
            return format;
        }
        if (tiling == vk::ImageTiling::eOptimal && (properties.optimalTilingFeatures & features) == features)
        {
            return format;
        }
    }
    return {};
}

auto Device::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const noexcept
    -> std::optional<uint32_t>
{
    const auto memoryProperties = physicalDevice.getMemoryProperties();

    for (auto i = uint32_t {}; i < memoryProperties.memoryTypeCount; i++)
    {
        if (((typeFilter & (uint32_t {1} << i)) != 0) &&
            (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }
    return {};
}

auto Device::QueueFamilies::getUniqueQueueFamilies() const -> std::unordered_set<uint32_t>
{
    return std::unordered_set {graphicsFamily, presentationFamily};
}

}
