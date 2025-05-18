#pragma once

#include <panda/gfx/vulkan/Context.h>
#include <panda/gfx/vulkan/Scene.h>

#include <cuda/Simulation.cuh>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/ext/vector_uint3.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Window.hpp"
#include "cuda/refinement/RefinementParameters.cuh"
#include "utils/ConfigurationManager.hpp"

namespace sph
{

class App
{
public:
    explicit App(std::string configPath = "config.json");
    auto run() -> int;
    auto runBenchmarks() -> void;

private:
    static auto initializeLogger() -> void;
    static auto registerSignalHandlers() -> void;
    static auto calculateParticleSpacing(const glm::vec3& domainSize, const glm::uvec3& gridSize) -> glm::vec3;

    auto loadConfigurationFromFile(const std::string& configPath) -> bool;

    auto mainLoop() const -> void;
    auto setDefaultScene() -> void;

    void createDomainBoundaries();
    void createParticleDistribution();
    void setupLighting() const;

    void createParticlesInGrid(const glm::vec3& startPos, const glm::uvec3& gridSize, const glm::vec3& spacing);

    std::vector<glm::vec4> _particles;
    std::unique_ptr<panda::gfx::vulkan::Scene> _scene;

    std::unique_ptr<Window> _window;
    std::unique_ptr<panda::gfx::vulkan::Context> _api;
    std::unique_ptr<cuda::Simulation> _simulation;
    cuda::Simulation::Parameters _simulationParameters {};
    cuda::refinement::RefinementParameters _refinementParameters {};
    InitialParameters _initialParameters {};
    std::optional<BenchmarkParameters> _benchmarkParams;

    ConfigurationManager _configManager;
    std::string _configPath;
};
}
