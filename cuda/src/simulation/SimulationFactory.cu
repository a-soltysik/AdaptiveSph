#include <glm/ext/vector_float4.hpp>
#include <memory>
#include <optional>
#include <vector>

#include "../../include/cuda/simulation/Simulation.cuh"
#include "AdaptiveSphSimulation.cuh"
#include "SphSimulation.cuh"
#include "cuda/refinement/RefinementParameters.cuh"

namespace sph::cuda
{

auto createSimulation(const Simulation::Parameters& parameters,
                      const std::vector<glm::vec4>& positions,
                      const FluidParticlesDataImportedBuffer& fluidParticleMemory,
                      const BoundaryParticlesDataImportedBuffer& boundaryParticleMemory,
                      const physics::StaticBoundaryDomain& boundaryDomain,
                      [[maybe_unused]] const std::optional<refinement::RefinementParameters>& refinementParams,
                      uint32_t maxFluidParticleCapacity,
                      uint32_t maxBoundaryParticleCapacity) -> std::unique_ptr<Simulation>
{
    if (refinementParams.has_value() && refinementParams->enabled == true)
    {
        return std::make_unique<AdaptiveSphSimulation>(parameters,
                                                       positions,
                                                       fluidParticleMemory,
                                                       boundaryParticleMemory,
                                                       boundaryDomain,
                                                       refinementParams.value(),
                                                       maxBoundaryParticleCapacity);
    }
    return std::make_unique<SphSimulation>(parameters,
                                           positions,
                                           fluidParticleMemory,
                                           boundaryParticleMemory,
                                           boundaryDomain,
                                           maxFluidParticleCapacity,
                                           maxBoundaryParticleCapacity);
}

}