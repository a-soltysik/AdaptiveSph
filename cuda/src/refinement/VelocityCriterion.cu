#include <cfloat>
#include <glm/ext/quaternion_geometric.hpp>
#include <glm/ext/vector_float3.hpp>

#include "VelocityCriterion.cuh"

namespace sph::cuda::refinement::velocity
{

SplitCriterionGenerator::SplitCriterionGenerator(float minimalMass, VelocityParameters::Split split)
    : _minimalMass {minimalMass},
      _split {split}
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
    if (velocityMagnitude < _split.minimalSpeedThreshold)
    {
        return FLT_MAX;
    }
    return velocityMagnitude;
}

MergeCriterionGenerator::MergeCriterionGenerator(float maximalMass, VelocityParameters::Merge merge)
    : _maximalMass {maximalMass},
      _merge {merge}
{
}

__device__ auto MergeCriterionGenerator::operator()(ParticlesData particles, uint32_t id) const -> float
{
    if (particles.masses[id] > _maximalMass)
    {
        return FLT_MAX;
    }
    const auto velocity = particles.velocities[id];
    const auto velocityMagnitude = glm::length(glm::vec3(velocity));
    if (velocityMagnitude > _merge.maximalSpeedThreshold)
    {
        return FLT_MAX;
    }
    return velocityMagnitude;
}

}
