#pragma once
#include <cstdint>

#include "cuda/Simulation.cuh"
#include "cuda/refinement/RefinementParameters.cuh"
#include "simulation/adaptive/SphSimulation.cuh"

namespace sph::cuda::refinement::velocity
{

class SplitCriterionGenerator
{
public:
    __host__ __device__ SplitCriterionGenerator(float minimalMass, VelocityParameters::Split split)
        : _minimalMass(minimalMass),
          _split(split)
    {
    }

    __device__ auto operator()(ParticlesData particles,
                               uint32_t id,
                               const SphSimulation::Grid& grid,
                               const Simulation::Parameters& simulationData) const -> float;

private:
    float _minimalMass;
    VelocityParameters::Split _split;
};

class MergeCriterionGenerator
{
public:
    __host__ __device__ MergeCriterionGenerator(float maximalMass, VelocityParameters::Merge merge)
        : _maximalMass(maximalMass),
          _merge(merge)
    {
    }

    __device__ auto operator()(ParticlesData particles,
                               uint32_t id,
                               const SphSimulation::Grid& grid,
                               const Simulation::Parameters& simulationData) const -> float;

private:
    float _maximalMass;
    VelocityParameters::Merge _merge;
};

}
