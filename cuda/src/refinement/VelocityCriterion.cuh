#pragma once
#include "cuda/Simulation.cuh"

namespace sph::cuda::refinement::velocity
{

class SplitCriterionGenerator
{
public:
    SplitCriterionGenerator(float minimalMass, float minimalVelocity);
    __device__ auto operator()(ParticlesData particles, uint32_t id) const -> float;

private:
    float _minimalMass;
    float _minimalVelocity;
};

class MergeCriterionGenerator
{
public:
    MergeCriterionGenerator(float maximalMass, float miximalVelocity);
    __device__ auto operator()(ParticlesData particles, uint32_t id) const -> float;

private:
    float _maximalMass;
    float _maximalVelocity;
};

}
