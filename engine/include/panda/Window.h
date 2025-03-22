#pragma once

// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include <vulkan/vulkan_core.h>

#include <glm/ext/vector_uint2.hpp>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace panda
{

class Window
{
public:
    using Id = size_t;

    virtual ~Window() noexcept = default;

    [[nodiscard]] virtual auto shouldClose() const -> bool = 0;
    [[nodiscard]] virtual auto isMinimized() const -> bool = 0;
    [[nodiscard]] virtual auto getSize() const -> glm::uvec2 = 0;
    [[nodiscard]] virtual auto getRequiredExtensions() const -> std::vector<const char*> = 0;
    [[nodiscard]] virtual auto createSurface(VkInstance instance) const -> VkSurfaceKHR = 0;
    [[nodiscard]] virtual auto getId() const -> Id = 0;
    virtual auto processInput() -> void = 0;
    virtual auto waitForInput() -> void = 0;
};

}
