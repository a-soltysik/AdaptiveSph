#pragma once
#include "ExperimentBase.hpp"

namespace sph::benchmark
{

class LidDrivenCavity : public ExperimentBase
{
public:
    LidDrivenCavity();

    // Run the lid-driven cavity benchmark
    BenchmarkResult runBenchmark(const BenchmarkParameters& params,
                                 const cuda::Simulation::Parameters& simulationParameters,
                                 BenchmarkResult::SimulationType simulationType,
                                 panda::gfx::vulkan::Context& api,
                                 bool visualize = true,
                                 Window* window = nullptr) override;  // Add visualization flag

protected:
    // Create simulation parameters for lid-driven cavity
    cuda::Simulation::Parameters createSimulationParameters(const BenchmarkParameters& params,
                                                            const cuda::Simulation::Parameters& simulationParameters,
                                                            BenchmarkResult::SimulationType simulationType) override;

    // Initialize particles for lid-driven cavity
    void initializeParticles(std::vector<glm::vec4>& particles,
                             const cuda::Simulation::Parameters& simulationParams) override;
};

}  // namespace sph::benchmark
