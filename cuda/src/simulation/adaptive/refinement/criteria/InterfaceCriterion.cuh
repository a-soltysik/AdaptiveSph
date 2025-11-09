#pragma once
#include "cuda/Simulation.cuh"
#include "cuda/refinement/RefinementParameters.cuh"
#include "glm/gtx/component_wise.hpp"
#include "simulation/SphSimulation.cuh"

namespace sph::cuda::refinement::interfaceCriterion
{

class SplitCriterionGenerator
{
public:
    __host__ __device__ SplitCriterionGenerator(float minimalMass, InterfaceParameters interfaceParameters)
        : _minimalMass(minimalMass),
          _interface(interfaceParameters)
    {
    }

    __device__ auto operator()(ParticlesData particles,
                               uint32_t id,
                               const SphSimulation::Grid& grid,
                               const Simulation::Parameters& simulationData) const -> float;

private:
    float _minimalMass;
    InterfaceParameters _interface;
};

class MergeCriterionGenerator
{
public:
    __host__ __device__ MergeCriterionGenerator(float maximalMass, InterfaceParameters interfaceParameters)
        : _maximalMass(maximalMass),
          _interface(interfaceParameters)
    {
    }

    __device__ auto operator()(ParticlesData particles,
                               uint32_t id,
                               const SphSimulation::Grid& grid,
                               const Simulation::Parameters& simulationData) const -> float;

private:
    float _maximalMass;
    InterfaceParameters _interface;
};

inline __device__ auto calculateMinSurfaceDistance(const glm::vec4& position,
                                                   const Simulation::Parameters::Domain& domain) -> glm::vec3
{
    const auto distToMinX = position.x - domain.min.x;
    const auto distToMaxX = domain.max.x - position.x;
    const auto distToMinY = position.y - domain.min.y;
    const auto distToMaxY = domain.max.y - position.y;
    const auto distToMinZ = position.z - domain.min.z;
    const auto distToMaxZ = domain.max.z - position.z;

    const auto minDistX = glm::min(distToMinX, distToMaxX);
    const auto minDistY = glm::min(distToMinY, distToMaxY);
    const auto minDistZ = glm::min(distToMinZ, distToMaxZ);

    return {glm::max(0.0F, minDistX), glm::max(0.0F, minDistY), glm::max(0.0F, minDistZ)};
}

inline __device__ auto SplitCriterionGenerator::operator()(ParticlesData particles,
                                                           uint32_t id,
                                                           const SphSimulation::Grid& grid,
                                                           const Simulation::Parameters& simulationData) const -> float
{
    if (particles.masses[id] < _minimalMass)
    {
        return -1.0F;
    }

    const auto position = particles.positions[id];
    const auto minDistances = calculateMinSurfaceDistance(position, simulationData.domain);
    const auto domainSize = simulationData.domain.max - simulationData.domain.min;
    const auto splitThresholds = _interface.split.distanceRatioThreshold * domainSize;
    const auto normalizedDistances = minDistances / splitThresholds;
    const auto minNormalizedDistance = glm::compMin(normalizedDistances);

    return 1.0F - minNormalizedDistance;
}

inline __device__ auto MergeCriterionGenerator::operator()(ParticlesData particles,
                                                           uint32_t id,
                                                           const SphSimulation::Grid& grid,
                                                           const Simulation::Parameters& simulationData) const -> float
{
    if (particles.masses[id] > _maximalMass)
    {
        return -1.0F;
    }

    const auto position = particles.positions[id];
    const auto minDistances = calculateMinSurfaceDistance(position, simulationData.domain);
    const auto domainSize = simulationData.domain.max - simulationData.domain.min;
    const auto mergeThresholds = _interface.merge.distanceRatioThreshold * domainSize;
    const auto normalizedDistances = minDistances / mergeThresholds;
    const auto minNormalizedDistance = glm::compMin(normalizedDistances);

    return minNormalizedDistance - 1.0F;
}

}
