#pragma once
#include "ExperimentBase.hpp"

namespace sph::benchmark
{

class TaylorGreenVortex : public ExperimentBase
{
public:
    TaylorGreenVortex();

    // Run the Taylor-Green vortex benchmark
    BenchmarkResult runBenchmark(const BenchmarkParameters& params,
                                 const cuda::Simulation::Parameters& simulationParameters,
                                 BenchmarkResult::SimulationType simulationType,
                                 panda::gfx::vulkan::Context& api,
                                 bool visualize = true,
                                 Window* window = nullptr) override;

protected:
    // Create simulation parameters for Taylor-Green vortex
    cuda::Simulation::Parameters createSimulationParameters(const BenchmarkParameters& params,
                                                            const cuda::Simulation::Parameters& simulationParameters,
                                                            BenchmarkResult::SimulationType simulationType) override;

    // Initialize particles for Taylor-Green vortex
    void initializeParticles(std::vector<glm::vec4>& particles,
                             const cuda::Simulation::Parameters& simulationParams) override;
};

}  // namespace sph::benchmark
