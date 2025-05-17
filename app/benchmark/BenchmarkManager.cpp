#include "BenchmarkManager.hpp"

#include <panda/Logger.h>

#include <filesystem>

#include "LidDrivenCavity.hpp"
#include "PoiseuilleFlow.hpp"
#include "TaylorGreenVortex.hpp"

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
    if (!experiment)
    {
        panda::log::Error("No experiment found with name: {}", std::to_underlying(params.testCase));
        return;
    }
    panda::log::Info("Running {} benchmark", std::to_underlying(experiment->getName()));
    MetricsCollector metricsCollector;
    static constexpr auto simulationTypes = std::array {//BenchmarkResult::SimulationType::Coarse,
                                                        //BenchmarkResult::SimulationType::Fine,
                                                        BenchmarkResult::SimulationType::Adaptive};
    for (const auto& simulationType : simulationTypes)
    {
        panda::log::Info("Running {} simulation", std::to_underlying(simulationType));

        auto result = experiment->runBenchmark(params, simulationParameters, simulationType, api, true, &window);

        metricsCollector.saveToFile(result, params.outputPath);
    }

    panda::log::Info("Benchmark completed for {}", std::to_underlying(experiment->getName()));
}

void BenchmarkManager::ensureOutputDirectoryExists(const std::string& outputPath)
{
    std::filesystem::path path(outputPath);
    if (!std::filesystem::exists(path))
    {
        std::filesystem::create_directories(path);
        panda::log::Info("Created output directory: {}", outputPath);
    }
}

ExperimentBase* BenchmarkManager::findExperimentByName(cuda::Simulation::Parameters::TestCase testCase) const
{
    for (const auto& experiment : _experiments)
    {
        if (experiment->getName() == testCase)
        {
            return experiment.get();
        }
    }
    return nullptr;
}

void BenchmarkManager::registerExperiment(std::unique_ptr<ExperimentBase> experiment)
{
    panda::log::Info("Registering benchmark experiment: {}", std::to_underlying(experiment->getName()));
    _experiments.push_back(std::move(experiment));
}

}  // namespace sph::benchmark
