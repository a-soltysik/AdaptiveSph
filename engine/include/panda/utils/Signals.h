#pragma once

// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include <functional>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "Signal.h"
#include "panda/Window.h"
#include "panda/gfx/Scene.h"

namespace panda::utils::signals
{
struct FrameBufferResizedData
{
    Window::Id id;
    int x;
    int y;
};

using FrameBufferResized = Signal<FrameBufferResizedData>;

struct BeginGuiRenderData
{
    vk::CommandBuffer commandBuffer;
    std::reference_wrapper<gfx::Scene> scene;
};

using BeginGuiRender = Signal<BeginGuiRenderData>;

using GameLoopIterationStarted = Signal<>;

inline auto frameBufferResized = FrameBufferResized {};
inline auto beginGuiRender = BeginGuiRender {};
inline auto gameLoopIterationStarted = GameLoopIterationStarted {};

}