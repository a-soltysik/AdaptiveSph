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
    static auto getInitialSimulationParameters(const cuda::Simulation::Parameters::Domain& domain,
                                               uint32_t particleCount,
                                               float totalMass) -> cuda::Simulation::Parameters;
    static auto getInitialRefinementParameters() -> cuda::refinement::RefinementParameters;

    auto mainLoop() const -> void;
    auto setDefaultScene() -> void;

    std::vector<glm::vec4> _particles;
    std::unique_ptr<panda::gfx::vulkan::Scene> _scene;

    std::unique_ptr<Window> _window;
    std::unique_ptr<panda::gfx::vulkan::Context> _api;
    std::unique_ptr<cuda::Simulation> _simulation;
    cuda::Simulation::Parameters _simulationParameters {};
    cuda::refinement::RefinementParameters _refinementParameters {};
};
}
