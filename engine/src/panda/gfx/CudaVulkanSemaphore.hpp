#pragma once

#include <cuda/memory/ImportedSemaphore.cuh>
#include <memory>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_handles.hpp>

namespace panda::gfx
{

class Device;

class CudaVulkanSemaphore final
{
public:
    explicit CudaVulkanSemaphore(const Device& device);
    CudaVulkanSemaphore(const CudaVulkanSemaphore&) = delete;
    auto operator=(const CudaVulkanSemaphore&) -> CudaVulkanSemaphore& = delete;
    CudaVulkanSemaphore(CudaVulkanSemaphore&&) = delete;
    auto operator=(CudaVulkanSemaphore&&) -> CudaVulkanSemaphore& = delete;
    ~CudaVulkanSemaphore() noexcept;

    [[nodiscard]] auto getHandle() const noexcept -> vk::Semaphore;
    auto waitCuda() const -> void;
    auto signalCuda() const -> void;

private:
    const Device& _device;
    vk::Semaphore _semaphore {};
    std::unique_ptr<sph::cuda::ImportedSemaphore> _cudaSemaphore;
};

}
