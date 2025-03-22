#pragma once

#include <panda/gfx/vulkan/Context.h>
#include <panda/gfx/vulkan/Scene.h>

#include <cuda/Simulation.cuh>
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
    static auto showFpsOverlay() -> void;

    auto mainLoop() const -> void;
    auto setDefaultScene() -> void;

    std::vector<glm::vec3*> _particles;
    std::unique_ptr<panda::gfx::vulkan::Scene> _scene;

    std::unique_ptr<Window> _window;
    std::unique_ptr<panda::gfx::vulkan::Context> _api;
    std::unique_ptr<sph::cuda::Simulation> _simulation;
};

}
