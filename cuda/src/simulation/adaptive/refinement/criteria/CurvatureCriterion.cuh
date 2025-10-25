#pragma once
#include <cstdint>

#include "../../../SphSimulation.cuh"
#include "cuda/Simulation.cuh"
#include "cuda/refinement/RefinementParameters.cuh"

namespace sph::cuda::refinement::curvature
{

class SplitCriterionGenerator
{
public:
    __host__ __device__ SplitCriterionGenerator(float minimalMass, CurvatureParameters::Split split)
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
    CurvatureParameters::Split _split;
};

class MergeCriterionGenerator
{
public:
    __host__ __device__ MergeCriterionGenerator(float maximalMass, CurvatureParameters::Merge merge)
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
    CurvatureParameters::Merge _merge;
};

}
