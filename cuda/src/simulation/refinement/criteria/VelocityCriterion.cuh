#pragma once
#include <cstdint>

#include "Common.cuh"
#include "cuda/refinement/RefinementParameters.cuh"
#include "cuda/simulation/Simulation.cuh"
#include "simulation/SphSimulation.cuh"

namespace sph::cuda::refinement::velocity
{

class SplitCriterionGenerator
{
public:
    __host__ __device__ SplitCriterionGenerator(float minimalMass,
                                                VelocityParameters::Split split,
                                                SplittingParameters splitting)
        : _minimalMass(minimalMass),
          _split(split),
          _splitting(splitting)
    {
    }

    __device__ auto operator()(FluidParticlesData particles,
                               uint32_t id,
                               const NeighborGrid::Device& grid,
                               const Simulation::Parameters& simulationData) const -> float;

private:
    float _minimalMass;
    VelocityParameters::Split _split;
    SplittingParameters _splitting;
};

class MergeCriterionGenerator
{
public:
    __host__ __device__ MergeCriterionGenerator(float maximalMass, VelocityParameters::Merge merge)
        : _maximalMass(maximalMass),
          _merge(merge)
    {
    }

    __device__ auto operator()(FluidParticlesData particles,
                               uint32_t id,
                               const NeighborGrid::Device& grid,
                               const Simulation::Parameters& simulationData) const -> float;

private:
    float _maximalMass;
    VelocityParameters::Merge _merge;
};

inline __device__ auto SplitCriterionGenerator::operator()(FluidParticlesData particles,
                                                           uint32_t id,
                                                           const NeighborGrid::Device& grid,
                                                           const Simulation::Parameters& simulationData) const -> float
{
    if (particles.masses[id] < _minimalMass)
    {
        return -1;
    }

    const auto smoothingRadius = particles.smoothingRadii[id];
    if (!checkIfParticleCanBeSplitInsideDomain(simulationData.domain,
                                               particles.positions[id],
                                               smoothingRadius,
                                               _splitting.epsilon))
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

inline __device__ auto MergeCriterionGenerator::operator()(FluidParticlesData particles,
                                                           uint32_t id,
                                                           const NeighborGrid::Device& grid,
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
