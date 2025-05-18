#pragma once
#include "../../SphSimulation.cuh"
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

}
