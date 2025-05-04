#include <glm/ext/vector_float4.hpp>
#include <memory>
#include <vector>

#include "AdaptiveSphSimulation.cuh"
#include "SphSimulation.cuh"
#include "cuda/Simulation.cuh"
#include "cuda/refinement/RefinementParameters.cuh"

namespace sph::cuda
{

auto createSimulation(const Simulation::Parameters& parameters,
                      const std::vector<glm::vec4>& positions,
                      const ParticlesDataBuffer& memory,
                      const refinement::RefinementParameters& refinementParams) -> std::unique_ptr<Simulation>
{
    if (refinementParams.enabled)
    {
        return std::make_unique<AdaptiveSphSimulation>(parameters, positions, memory, refinementParams);
    }
    return std::make_unique<SphSimulation>(parameters, positions, memory, refinementParams.maxParticleCount);
}

}