// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include "CudaVulkanSemaphore.hpp"

#include <vulkan/vulkan.h>  // NOLINT(misc-include-cleaner)

#include <cuda/memory/ImportedSemaphore.cuh>
#include <memory>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "panda/gfx/Device.h"
#include "panda/utils/format/gfx/api/ResultFormatter.h"  // NOLINT(misc-include-cleaner)

namespace panda::gfx
{

CudaVulkanSemaphore::CudaVulkanSemaphore(const Device& device)
    : _device {device}
{
#ifdef WIN32
    static constexpr auto handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32;
#else
    static constexpr auto handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd;
#endif

    static constexpr auto exportInfo = vk::ExportSemaphoreCreateInfo {.handleTypes = handleType};
    const auto createInfo = vk::SemaphoreCreateInfo {.pNext = &exportInfo};
    _semaphore = expect(_device.logicalDevice.createSemaphore(createInfo),
                        vk::Result::eSuccess,
                        "Failed to create CUDA-Vulkan semaphore");

    try
    {
#ifdef WIN32
        const auto handleInfo = vk::SemaphoreGetWin32HandleInfoKHR {.semaphore = _semaphore, .handleType = handleType};
        const auto handle = expect(_device.logicalDevice.getSemaphoreWin32HandleKHR(handleInfo),
                                   vk::Result::eSuccess,
                                   "Failed to export CUDA-Vulkan semaphore handle");
        _cudaSemaphore = sph::cuda::importSemaphore(handle);
#else
        const auto handleInfo = vk::SemaphoreGetFdInfoKHR {.semaphore = _semaphore, .handleType = handleType};
        const auto handle = expect(_device.logicalDevice.getSemaphoreFdKHR(handleInfo),
                                   vk::Result::eSuccess,
                                   "Failed to export CUDA-Vulkan semaphore file descriptor");
        _cudaSemaphore = sph::cuda::importSemaphore(handle);
#endif
    }
    catch (...)
    {
        _device.logicalDevice.destroySemaphore(_semaphore);
        _semaphore = nullptr;
        throw;
    }
}

CudaVulkanSemaphore::~CudaVulkanSemaphore() noexcept
{
    _cudaSemaphore.reset();
    if (_semaphore)
    {
        _device.logicalDevice.destroySemaphore(_semaphore);
    }
}

auto CudaVulkanSemaphore::getHandle() const noexcept -> vk::Semaphore
{
    return _semaphore;
}

auto CudaVulkanSemaphore::waitCuda() const -> void
{
    _cudaSemaphore->wait();
}

auto CudaVulkanSemaphore::signalCuda() const -> void
{
    _cudaSemaphore->signal();
}

}
