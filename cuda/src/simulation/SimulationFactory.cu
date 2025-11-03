#include <glm/ext/vector_float4.hpp>
#include <memory>
#include <optional>
#include <vector>

#include "SphSimulation.cuh"
#include "adaptive/AdaptiveSphSimulation.cuh"
#include "cuda/Simulation.cuh"
#include "cuda/refinement/RefinementParameters.cuh"

namespace sph::cuda
{

auto createSimulation(const Simulation::Parameters& parameters,
                      const std::vector<glm::vec4>& positions,
                      const ParticlesDataBuffer& memory,
                      const std::optional<refinement::RefinementParameters>& refinementParams)
    -> std::unique_ptr<Simulation>
{
    if (refinementParams.has_value() && refinementParams->enabled == true)
    {
        return std::make_unique<AdaptiveSphSimulation>(parameters, positions, memory, refinementParams.value());
    }
    return std::make_unique<SphSimulation>(parameters, positions, memory, refinementParams.value().maxParticleCount);
}

}