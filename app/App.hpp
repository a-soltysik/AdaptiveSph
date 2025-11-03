#pragma once

#include <panda/gfx/Context.h>
#include <panda/gfx/Scene.h>

#include <cuda/Simulation.cuh>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/ext/vector_uint3.hpp>
#include <memory>
#include <string>
#include <vector>

#include "ui/Window.hpp"
#include "utils/Configuration.hpp"

namespace sph
{
class App
{
public:
    explicit App(std::string configPath = "config.json");

    auto run() -> int;

private:
    static auto initializeLogger() -> void;

    static auto registerSignalHandlers() -> void;

    static auto calculateParticleSpacing(const glm::vec3& domainSize, const glm::uvec3& gridSize) -> glm::vec3;

    static auto loadConfigurationFromFile(const std::string& configPath) -> utils::Configuration;

    auto mainLoop() const -> void;

    auto setDefaultScene(const cuda::Simulation::Parameters& simulationParameters,
                         const utils::InitialParameters& initialParameters) -> void;

    void createDomainBoundaries(cuda::Simulation::Parameters::Domain domain) const;

    void createParticleDistribution(const cuda::Simulation::Parameters& simulationParameters,
                                    const utils::InitialParameters& initialParameters);

    void setupLighting() const;

    void createParticlesInGrid(const glm::vec3& startPos,
                               const utils::InitialParameters& initialParameters,
                               const glm::vec3& spacing);

    std::vector<glm::vec4> _particles;
    std::unique_ptr<panda::gfx::Scene> _scene;
    std::unique_ptr<Window> _window;
    std::unique_ptr<panda::gfx::Context> _api;
    std::unique_ptr<cuda::Simulation> _simulation;

    std::string _configPath;
};
}
