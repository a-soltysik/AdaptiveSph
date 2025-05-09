#pragma once
#include "cuda/Simulation.cuh"
#include "cuda/refinement/VelocityParameters.cuh"

namespace sph::cuda::refinement::velocity
{

class SplitCriterionGenerator
{
public:
    SplitCriterionGenerator(float minimalMass, VelocityParameters::Split split);
    __device__ auto operator()(ParticlesData particles, uint32_t id) const -> float;

private:
    float _minimalMass;
    VelocityParameters::Split _split;
};

class MergeCriterionGenerator
{
public:
    MergeCriterionGenerator(float maximalMass, VelocityParameters::Merge merge);
    __device__ auto operator()(ParticlesData particles, uint32_t id) const -> float;

private:
    float _maximalMass;
    VelocityParameters::Merge _merge;
};

}
