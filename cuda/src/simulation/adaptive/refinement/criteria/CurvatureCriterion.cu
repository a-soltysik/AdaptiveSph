#include <cstdint>
#include <glm/common.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/geometric.hpp>

#include "../../../SphSimulation.cuh"
#include "CurvatureCriterion.cuh"
#include "cuda/Simulation.cuh"

namespace sph::cuda::refinement::curvature
{

__device__ auto SplitCriterionGenerator::operator()(ParticlesData particles,
                                                    uint32_t id,
                                                    const SphSimulation::Grid& grid,
                                                    const Simulation::Parameters& simulationData) const -> float
{
    if (particles.masses[id] < _minimalMass)
    {
        return -1.0F;
    }

    const auto velocity = glm::vec3(particles.velocities[id]);
    const auto velocityMagnitude = glm::length(velocity);
    if (velocityMagnitude < _split.minimalVelocityThreshold)
    {
        return -1.0F;
    }

    // Użyj już obliczonego przyspieszenia!
    const auto acceleration = glm::vec3(particles.accelerations[id]);

    // κ = |v × a| / |v|³
    const auto crossProduct = glm::cross(velocity, acceleration);
    const auto curvature = glm::length(crossProduct) / (velocityMagnitude * velocityMagnitude * velocityMagnitude);

    if (curvature > _split.minimalCurvatureThreshold)
    {
        return curvature;
    }

    return -1.0F;
}

__device__ auto MergeCriterionGenerator::operator()(ParticlesData particles,
                                                    uint32_t id,
                                                    const SphSimulation::Grid& grid,
                                                    const Simulation::Parameters& simulationData) const -> float
{
    if (particles.masses[id] > _maximalMass)
    {
        return -1.0F;
    }

    const auto velocity = glm::vec3(particles.velocities[id]);
    const auto velocityMagnitude = glm::length(velocity);

    if (velocityMagnitude < _merge.minimalVelocityThreshold)
    {
        return -1.0F;
    }

    // Użyj już obliczonego przyspieszenia!
    const auto acceleration = glm::vec3(particles.accelerations[id]);
    const auto crossProduct = glm::cross(velocity, acceleration);
    const auto curvature = glm::length(crossProduct) / (velocityMagnitude * velocityMagnitude * velocityMagnitude);

    if (curvature < _merge.maximalCurvatureThreshold)
    {
        return curvature / _merge.maximalCurvatureThreshold;
    }

    return -1.0F;
}

}
