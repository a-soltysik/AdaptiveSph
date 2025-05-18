#pragma once
#include <panda/gfx/vulkan/Context.h>

#include <memory>
#include <string>
#include <vector>

#include "../ui/Window.hpp"
#include "cuda/Simulation.cuh"
#include "experiments/ExperimentBase.hpp"
#include "utils/ConfigurationManager.hpp"

namespace sph::benchmark
{

class BenchmarkManager
{
public:
    BenchmarkManager();
    void runBenchmarks(const BenchmarkParameters& params,
                       const cuda::Simulation::Parameters& simulationParameters,
                       panda::gfx::vulkan::Context& api,
                       Window& window) const;

    void registerExperiment(std::unique_ptr<ExperimentBase> experiment);

private:
    static void ensureOutputDirectoryExists(const std::string& outputPath);
    [[nodiscard]] auto findExperimentByName(cuda::Simulation::Parameters::TestCase testCase) const -> ExperimentBase*;

    std::vector<std::unique_ptr<ExperimentBase>> _experiments;
};

}
