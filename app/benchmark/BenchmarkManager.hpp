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
    static void ensureOutputDirectoryExists(const std::string& outputPath);
    ExperimentBase* findExperimentByName(cuda::Simulation::Parameters::TestCase testCase) const;

    std::vector<std::unique_ptr<ExperimentBase>> _experiments;
};

}  // namespace sph::benchmark
