// InterfaceCriterion.cuh - Simplified header
#pragma once
#include "cuda/Simulation.cuh"
#include "cuda/refinement/RefinementParameters.cuh"

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

    __device__ auto operator()(ParticlesData particles, uint32_t id, const Simulation::Parameters& simulationData) const
        -> float;

private:
    __device__ float computePriorityScore(float distance) const;

    float _minimalMass;
    InterfaceParameters _interface;
    mutable glm::vec3 _splitThreshold;  // Calculated threshold stored for priority computation
};

class MergeCriterionGenerator
{
public:
    __host__ __device__ MergeCriterionGenerator(float maximalMass, InterfaceParameters interfaceParameters)
        : _maximalMass(maximalMass),
          _interface(interfaceParameters)
    {
    }

    __device__ auto operator()(ParticlesData particles, uint32_t id, const Simulation::Parameters& simulationData) const
        -> float;

private:
    __device__ float computePriorityScore(float distance) const;

    float _maximalMass;
    InterfaceParameters _interface;
    mutable glm::vec3 _mergeThreshold;  // Calculated threshold stored for priority computation
};

}
