#include <cfloat>

#include "VelocityCriterion.cuh"
#include "glm/detail/func_geometric.inl"

namespace sph::cuda::refinement::velocity
{

SplitCriterionGenerator::SplitCriterionGenerator(float minimalMass, float minimalVelocity)
    : _minimalMass {minimalMass},
      _minimalVelocity {minimalVelocity}
{
}

__device__ auto SplitCriterionGenerator::operator()(ParticlesData particles, uint32_t id) const -> float
{
    if (particles.masses[id] < _minimalMass)
    {
        return FLT_MAX;
    }
    const auto velocity = particles.velocities[id];
    const auto velocityMagnitude = glm::length(glm::vec3(velocity));
    if (velocityMagnitude < _minimalVelocity)
    {
        return FLT_MAX;
    }
    return velocityMagnitude;
}

MergeCriterionGenerator::MergeCriterionGenerator(float maximalMass, float maximalVelocity)
    : _maximalMass {maximalMass},
      _maximalVelocity {maximalVelocity}
{
}

__device__ auto MergeCriterionGenerator::operator()(ParticlesData particles, uint32_t id) const -> float
{
    if (particles.masses[id] > _maximalVelocity)
    {
        return FLT_MAX;
    }
    const auto velocity = particles.velocities[id];
    const auto velocityMagnitude = glm::length(glm::vec3(velocity));
    if (velocityMagnitude > _maximalVelocity)
    {
        return FLT_MAX;
    }
    return velocityMagnitude;
}

}
