#pragma once

#include <vulkan/vulkan_handles.hpp>

namespace panda::gfx
{

struct FrameSemaphores final
{
    vk::Semaphore cudaReady {};
    vk::Semaphore vulkanDone {};
};

}
