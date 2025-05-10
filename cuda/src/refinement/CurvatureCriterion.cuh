// CurvatureCriterion.cuh
#pragma once
#include "../SphSimulation.cuh"
#include "cuda/Simulation.cuh"
#include "cuda/refinement/RefinementParameters.cuh"

namespace sph::cuda::refinement::curvature
{

class SplitCriterionGenerator
{
public:
    __host__ __device__ SplitCriterionGenerator(float minimalMass, CurvatureParameters curvature)
        : _minimalMass(minimalMass),
          _curvature(curvature)
    {
    }

    __device__ auto operator()(ParticlesData particles,
                               uint32_t id,
                               const SphSimulation::Grid& grid,
                               const Simulation::Parameters& simulationData) const -> float;

private:
    float _minimalMass;
    CurvatureParameters _curvature;
};

class MergeCriterionGenerator
{
public:
    __host__ __device__ MergeCriterionGenerator(float maximalMass, CurvatureParameters curvature)
        : _maximalMass(maximalMass),
          _curvature(curvature)
    {
    }

    __device__ auto operator()(ParticlesData particles,
                               uint32_t id,
                               const SphSimulation::Grid& grid,
                               const Simulation::Parameters& simulationData) const -> float;

private:
    float _maximalMass;
    CurvatureParameters _curvature;
};

}
