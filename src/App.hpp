#pragma once

#include <panda/gfx/vulkan/Context.h>
#include <panda/gfx/vulkan/Scene.h>

#include <memory>

#include "Window.hpp"

namespace sph
{

class App
{
public:
    auto run() -> int;

private:
    static auto initializeLogger() -> void;
    static auto registerSignalHandlers() -> void;
    auto mainLoop() -> void;
    auto setDefaultScene() -> void;
    static auto showFpsOverlay() -> void;

    std::vector<glm::vec3*> _particles;
    panda::gfx::vulkan::Scene _scene {};

    std::unique_ptr<Window> _window;
    std::unique_ptr<panda::gfx::vulkan::Context> _api;
};

}
