
#pragma once
#include "ExperimentBase.hpp"

namespace sph::benchmark
{

class PoiseuilleFlow : public ExperimentBase
{
public:
    PoiseuilleFlow();

    // Run the Poiseuille flow benchmark
    BenchmarkResult runBenchmark(const BenchmarkParameters& params,
                                 const cuda::Simulation::Parameters& simulationParameters,
                                 BenchmarkResult::SimulationType simulationType,
                                 panda::gfx::vulkan::Context& api,
                                 bool visualize = true,
                                 Window* window = nullptr) override;

protected:
    // Create simulation parameters for Poiseuille flow
    cuda::Simulation::Parameters createSimulationParameters(const BenchmarkParameters& params,
                                                            const cuda::Simulation::Parameters& simulationParameters,
                                                            BenchmarkResult::SimulationType simulationType) override;

    // Initialize particles for Poiseuille flow
    void initializeParticles(std::vector<glm::vec4>& particles,
                             const cuda::Simulation::Parameters& simulationParams) override;
};

}  // namespace sph::benchmark
