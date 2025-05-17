// BenchmarkManager.hpp
#pragma once
#include <panda/gfx/vulkan/Context.h>

#include <memory>
#include <vector>

#include "ExperimentBase.hpp"

namespace sph::benchmark
{

class BenchmarkManager
{
public:
    BenchmarkManager();
    // Run all configured benchmark experiments
    void runBenchmarks(const BenchmarkParameters& params,
                       const cuda::Simulation::Parameters& simulationParameters,
                       panda::gfx::vulkan::Context& api,
                       Window& window) const;
    // Register a new experiment
    void registerExperiment(std::unique_ptr<ExperimentBase> experiment);

private:
    std::vector<std::unique_ptr<ExperimentBase>> _experiments;
};

}  // namespace sph::benchmark
