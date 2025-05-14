// PoiseuilleFlow.hpp
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
                                 BenchmarkResult::SimulationType simulationType,
                                 panda::gfx::vulkan::Context& api,
                                 bool visualize = true,
                                 Window* window = nullptr) override;

protected:
    // Create simulation parameters for Poiseuille flow
    cuda::Simulation::Parameters createSimulationParameters(const BenchmarkParameters& params,
                                                            BenchmarkResult::SimulationType simulationType) override;

    // Initialize particles for Poiseuille flow
    void initializeParticles(std::vector<glm::vec4>& particles,
                             const cuda::Simulation::Parameters& simulationParams) override;

    // Create refinement parameters
    cuda::refinement::RefinementParameters createRefinementParameters(
        const BenchmarkParameters& params,
        BenchmarkResult::SimulationType simulationType,
        const cuda::Simulation::Parameters& simulationParams);
};

}  // namespace sph::benchmark
