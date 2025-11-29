#pragma once

namespace sph::cuda::refinement
{

__device__ auto checkIfParticleCanBeSplitInsideDomain(Simulation::Parameters::Domain domain,
                                                      glm::vec4 position,
                                                      float smoothingRadius,
                                                      float epsilon) -> bool
{
    const auto maxOffset = epsilon * smoothingRadius;

    if (position.x - maxOffset < domain.min.x || position.x + maxOffset > domain.max.x ||
        position.y - maxOffset < domain.min.y || position.y + maxOffset > domain.max.y ||
        position.z - maxOffset < domain.min.z || position.z + maxOffset > domain.max.z)
    {
        return false;
    }
    return true;
}

}