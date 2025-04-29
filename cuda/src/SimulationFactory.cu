#include <memory>

#include "AdaptiveSphSimulation.cuh"
#include "SphSimulation.cuh"

namespace sph::cuda
{

/**
 * Factory function to create an appropriate SPH simulation instance
 * based on provided parameters
 */
std::unique_ptr<Simulation> createSimulation(const Simulation::Parameters& parameters,
                                             const std::vector<glm::vec4>& positions,
                                             const ParticlesDataBuffer& memory,
                                             const refinement::RefinementParameters& refinementParams)
{
    // Create adaptive simulation if refinement is enabled
    if (refinementParams.enabled)
    {
        return std::make_unique<AdaptiveSphSimulation>(parameters, positions, memory, refinementParams);
    }

    // Otherwise create standard simulation
    return std::make_unique<SphSimulation>(parameters, positions, memory, refinementParams.maxParticleCount);
}

}  // namespace sph::cuda
