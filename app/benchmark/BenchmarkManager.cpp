#include "BenchmarkManager.hpp"

#include <panda/Logger.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <memory>
#include <ranges>
#include <string>
#include <utility>

#include "Window.hpp"
#include "benchmark/ExperimentBase.hpp"
#include "benchmark/LidDrivenCavity.hpp"
#include "benchmark/MetricsCollector.hpp"
#include "benchmark/PoiseuilleFlow.hpp"
#include "benchmark/TaylorGreenVortex.hpp"
#include "cuda/Simulation.cuh"
#include "panda/gfx/vulkan/Context.h"
#include "utils/ConfigurationManager.hpp"

namespace sph::benchmark
{

BenchmarkManager::BenchmarkManager()
{
    registerExperiment(std::make_unique<LidDrivenCavity>());
    registerExperiment(std::make_unique<PoiseuilleFlow>());
    registerExperiment(std::make_unique<TaylorGreenVortex>());
}

void BenchmarkManager::runBenchmarks(const BenchmarkParameters& params,
                                     const cuda::Simulation::Parameters& simulationParameters,
                                     panda::gfx::vulkan::Context& api,
                                     Window& window) const
{
    if (!params.enabled)
    {
        panda::log::Info("Benchmarking is disabled in configuration");
        return;
    }
    panda::log::Info("Starting benchmark suite with {} experiments", _experiments.size());
    ensureOutputDirectoryExists(params.outputPath);
    auto* experiment = findExperimentByName(params.testCase);
    if (experiment == nullptr)
    {
        panda::log::Error("No experiment found with name: {}", std::to_underlying(params.testCase));
        return;
    }
    panda::log::Info("Running {} benchmark", std::to_underlying(experiment->getName()));
    static constexpr auto simulationTypes = std::array {BenchmarkResult::SimulationType::Coarse,
                                                        BenchmarkResult::SimulationType::Fine,
                                                        BenchmarkResult::SimulationType::Adaptive};
    for (const auto& simulationType : simulationTypes)
    {
        panda::log::Info("Running {} simulation", std::to_underlying(simulationType));

        auto result = experiment->runBenchmark(params, simulationParameters, simulationType, api, true, &window);

        sph::benchmark::MetricsCollector::saveToFile(result, params.outputPath);
    }

    panda::log::Info("Benchmark completed for {}", std::to_underlying(experiment->getName()));
}

void BenchmarkManager::ensureOutputDirectoryExists(const std::string& outputPath)
{
    const auto path = std::filesystem::path {outputPath};
    if (!std::filesystem::exists(path))
    {
        std::filesystem::create_directories(path);
        panda::log::Info("Created output directory: {}", outputPath);
    }
}

auto BenchmarkManager::findExperimentByName(cuda::Simulation::Parameters::TestCase testCase) const -> ExperimentBase*
{
    const auto it = std::ranges::find(_experiments, testCase, &ExperimentBase::getName);
    if (it == std::ranges::end(_experiments))
    {
        return nullptr;
    }
    return it->get();
}

void BenchmarkManager::registerExperiment(std::unique_ptr<ExperimentBase> experiment)
{
    panda::log::Info("Registering benchmark experiment: {}", std::to_underlying(experiment->getName()));
    _experiments.push_back(std::move(experiment));
}

}
