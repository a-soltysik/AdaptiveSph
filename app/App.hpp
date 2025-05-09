#pragma once

#include <panda/gfx/vulkan/Context.h>
#include <panda/gfx/vulkan/Scene.h>

#include <cuda/Simulation.cuh>
#include <memory>

#include "Window.hpp"
#include "utils/ConfigurationManager.hpp"

namespace sph
{

class App
{
public:
    explicit App(const std::string& configPath = "config.json");
    auto run() -> int;

private:
    static auto initializeLogger() -> void;
    static auto registerSignalHandlers() -> void;

    auto loadConfigurationFromFile(const std::string& configPath) -> bool;

    auto mainLoop() const -> void;
    auto setDefaultScene() -> void;

    std::vector<glm::vec4> _particles;
    std::unique_ptr<panda::gfx::vulkan::Scene> _scene;

    std::unique_ptr<Window> _window;
    std::unique_ptr<panda::gfx::vulkan::Context> _api;
    std::unique_ptr<cuda::Simulation> _simulation;
    cuda::Simulation::Parameters _simulationParameters {};
    cuda::refinement::RefinementParameters _refinementParameters {};
    InitialParameters _initialParameters {};

    ConfigurationManager _configManager;
    std::string _configPath;
};
}
