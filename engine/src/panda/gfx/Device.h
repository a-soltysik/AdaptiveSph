#pragma once

// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include <cstdint>
#include <optional>
#include <span>
#include <unordered_set>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "panda/Common.h"

namespace panda::gfx
{

class Device
{
public:
    struct QueueFamilies
    {
        uint32_t graphicsFamily;
        uint32_t presentationFamily;

        [[nodiscard]] auto getUniqueQueueFamilies() const -> std::unordered_set<uint32_t>;
    };

    struct SwapChainSupportDetails
    {
        vk::SurfaceCapabilitiesKHR capabilities;
        std::vector<vk::SurfaceFormatKHR> formats;
        std::vector<vk::PresentModeKHR> presentationModes;
    };

    Device(const vk::Instance& instance,
           const vk::SurfaceKHR& surface,
           std::span<const char* const> requiredExtensions);

    PD_DELETE_ALL(Device);

    ~Device() noexcept;

    [[nodiscard]] auto querySwapChainSupport() const -> SwapChainSupportDetails;
    [[nodiscard]] auto findSupportedFormat(std::span<const vk::Format> candidates,
                                           vk::ImageTiling tiling,
                                           vk::FormatFeatureFlags features) const noexcept -> std::optional<vk::Format>;
    [[nodiscard]] auto findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const noexcept
        -> std::optional<uint32_t>;

    const vk::PhysicalDevice physicalDevice;
    const QueueFamilies queueFamilies;

    const vk::Device logicalDevice;
    const vk::Queue graphicsQueue;
    const vk::Queue presentationQueue;
    const vk::CommandPool commandPool;

private:
    static auto pickPhysicalDevice(const vk::Instance& instance,
                                   const vk::SurfaceKHR& surface,
                                   std::span<const char* const> requiredExtensions) -> vk::PhysicalDevice;
    static auto isDeviceSuitable(vk::PhysicalDevice device,
                                 vk::SurfaceKHR surface,
                                 std::span<const char* const> requiredExtensions) -> bool;
    static auto findQueueFamilies(vk::PhysicalDevice device, vk::SurfaceKHR surface) -> std::optional<QueueFamilies>;
    static auto querySwapChainSupport(vk::PhysicalDevice device, vk::SurfaceKHR surface) -> SwapChainSupportDetails;
    static auto checkDeviceExtensionSupport(vk::PhysicalDevice device, std::span<const char* const> requiredExtensions)
        -> bool;
    static auto createLogicalDevice(vk::PhysicalDevice device,
                                    const QueueFamilies& queueFamilies,
                                    std::span<const char* const> requiredExtensions) -> vk::Device;

    const vk::SurfaceKHR& _surface;
};

}
