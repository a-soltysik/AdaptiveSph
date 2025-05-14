// BenchmarkManager.cpp
#include "BenchmarkManager.hpp"

#include <panda/Logger.h>

#include <filesystem>

#include "LidDrivenCavity.hpp"

namespace sph::benchmark
{

BenchmarkManager::BenchmarkManager()
{
    // Register default experiments
    registerExperiment(std::make_unique<LidDrivenCavity>());
}

void BenchmarkManager::runBenchmarks(const BenchmarkParameters& params,
                                     panda::gfx::vulkan::Context& api,
                                     Window& window)
{
    if (!params.enabled)
    {
        panda::log::Info("Benchmarking is disabled in configuration");
        return;
    }
    panda::log::Info("Starting benchmark suite with {} experiments", _experiments.size());
    // Create output directory if it doesn't exist
    std::filesystem::path outputPath(params.outputPath);
    if (!std::filesystem::exists(outputPath))
    {
        std::filesystem::create_directories(outputPath);
        panda::log::Info("Created output directory: {}", params.outputPath);
    }
    // Find the requested experiment
    for (const auto& experiment : _experiments)
    {
        if (experiment->getName() == params.testCase)
        {
            panda::log::Info("Running {} benchmark", std::to_underlying(experiment->getName()));
            MetricsCollector metricsCollector;
            // Run coarse simulation
            //panda::log::Info("Running coarse simulation");
            //auto coarseResult =
            //    experiment->runBenchmark(params, BenchmarkResult::SimulationType::Coarse, api, true, &window);
            //metricsCollector.saveToFile(coarseResult, params.outputPath);
            // Run fine simulation
            //panda::log::Info("Running fine simulation");
            //auto fineResult =
            //    experiment->runBenchmark(params, BenchmarkResult::SimulationType::Fine, api, true, &window);
            //metricsCollector.saveToFile(fineResult, params.outputPath);
            // Run adaptive simulation
            panda::log::Info("Running adaptive simulation");
            auto adaptiveResult =
                experiment->runBenchmark(params, BenchmarkResult::SimulationType::Adaptive, api, true, &window);
            metricsCollector.saveToFile(adaptiveResult, params.outputPath);
            panda::log::Info("Benchmark completed for {}", std::to_underlying(experiment->getName()));
            return;
        }
    }

    panda::log::Error("No experiment found with name: {}", std::to_underlying(params.testCase));
}

void BenchmarkManager::registerExperiment(std::unique_ptr<ExperimentBase> experiment)
{
    panda::log::Info("Registering benchmark experiment: {}", std::to_underlying(experiment->getName()));
    _experiments.push_back(std::move(experiment));
}

}  // namespace sph::benchmark
