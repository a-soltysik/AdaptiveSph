#include <cstdint>
#include <glm/ext/quaternion_geometric.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/geometric.hpp>

#include "VelocityCriterion.cuh"
#include "cuda/Simulation.cuh"
#include "simulation/adaptive/SphSimulation.cuh"

namespace sph::cuda::refinement::velocity
{

__device__ auto SplitCriterionGenerator::operator()(ParticlesData particles,
                                                    uint32_t id,
                                                    const SphSimulation::Grid& grid,
                                                    const Simulation::Parameters& simulationData) const -> float
{
    if (particles.masses[id] < _minimalMass)
    {
        return -1;
    }
    const auto velocity = particles.velocities[id];
    const auto velocityMagnitude = glm::length(glm::vec3(velocity));
    if (velocityMagnitude < _split.minimalSpeedThreshold)
    {
        return -1;
    }
    return velocityMagnitude;
}

__device__ auto MergeCriterionGenerator::operator()(ParticlesData particles,
                                                    uint32_t id,
                                                    const SphSimulation::Grid& grid,
                                                    const Simulation::Parameters& simulationData) const -> float
{
    if (particles.masses[id] > _maximalMass)
    {
        return -1;
    }

    const auto velocity = particles.velocities[id];
    const auto velocityMagnitude = glm::length(glm::vec3(velocity));

    if (velocityMagnitude < _merge.maximalSpeedThreshold)
    {
        return velocityMagnitude / _merge.maximalSpeedThreshold;
    }

    return -1;
}

}
