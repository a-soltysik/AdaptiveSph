// BenchmarkVisualizer.hpp
#pragma once
#include <panda/gfx/vulkan/Context.h>
#include <panda/gfx/vulkan/Scene.h>

#include <cuda/Simulation.cuh>
#include <memory>
#include <string>

#include "MetricsCollector.hpp"

namespace sph::benchmark
{

class BenchmarkVisualizer
{
public:
    BenchmarkVisualizer(panda::gfx::vulkan::Context& api,
                        cuda::Simulation::Parameters::TestCase experimentName,
                        BenchmarkResult::SimulationType simulationType);
    // Initialize visualization with specific simulation parameters
    void initialize(const cuda::Simulation::Parameters& params);

    // Update visualization for current frame
    void renderFrame(cuda::Simulation& simulation, float deltaTime);

private:
    void setupScene(const cuda::Simulation::Parameters& params);
    void updateCamera();

    panda::gfx::vulkan::Context& _api;
    std::unique_ptr<panda::gfx::vulkan::Scene> _scene;
    cuda::Simulation::Parameters::TestCase _experimentName;
    BenchmarkResult::SimulationType _simulationType;
    float _simulationTime = 0.0f;
};

}  // namespace sph::benchmark
