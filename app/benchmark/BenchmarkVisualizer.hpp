#pragma once
#include <panda/gfx/vulkan/Context.h>
#include <panda/gfx/vulkan/Scene.h>

#include <cuda/Simulation.cuh>
#include <memory>

#include "MetricsCollector.hpp"

namespace sph::benchmark
{

class BenchmarkVisualizer
{
public:
    BenchmarkVisualizer(panda::gfx::vulkan::Context& api,
                        cuda::Simulation::Parameters::TestCase experimentName,
                        BenchmarkResult::SimulationType simulationType);

    void initialize(const cuda::Simulation::Parameters& params);
    void renderFrame(const cuda::Simulation& simulation);

private:
    void setupScene(const cuda::Simulation::Parameters& params);
    void updateCamera();

    panda::gfx::vulkan::Context& _api;
    panda::gfx::vulkan::Scene _scene {};
    cuda::Simulation::Parameters::TestCase _experimentName;
    BenchmarkResult::SimulationType _simulationType;
};

}
