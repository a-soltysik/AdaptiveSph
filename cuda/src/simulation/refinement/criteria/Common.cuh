#pragma once

namespace sph::cuda::refinement
{

__device__ auto checkIfParticleCanBeSplitInsideDomain(Simulation::Parameters::Domain domain,
                                                      glm::vec4 position,
                                                      float smoothingRadius,
                                                      float epsilon) -> bool
{
    const auto maxOffset = epsilon * smoothingRadius * 2.F;

    //if (position.x - maxOffset < domain.min.x || position.x + maxOffset > domain.max.x)
    //{
    //    return false;
    //}
    return true;
    //return true;
}

}