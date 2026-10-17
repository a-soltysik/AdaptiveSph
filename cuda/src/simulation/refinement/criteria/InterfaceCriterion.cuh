#pragma once
#include "Common.cuh"
#include "algorithm/WendlandKernel.cuh"
#include "cuda/refinement/RefinementParameters.cuh"
#include "cuda/simulation/Simulation.cuh"
#include "simulation/SphSimulation.cuh"

namespace sph::cuda::refinement::interfaceCriterion
{

class SplitCriterionGenerator
{
public:
    __host__ __device__ SplitCriterionGenerator(float minimalMass,
                                                InterfaceParameters interfaceParameters,
                                                SplittingParameters splitting)
        : _minimalMass(minimalMass),
          _interface(interfaceParameters),
          _splitting(splitting)
    {
    }

    __device__ auto operator()(FluidParticlesData particles,
                               uint32_t id,
                               const NeighborGrid::Device& grid,
                               const Simulation::Parameters& simulationData) const -> float;

private:
    float _minimalMass;
    InterfaceParameters _interface;
    SplittingParameters _splitting;
};

class MergeCriterionGenerator
{
public:
    __host__ __device__ MergeCriterionGenerator(float maximalMass, InterfaceParameters interfaceParameters)
        : _maximalMass(maximalMass),
          _interface(interfaceParameters)
    {
    }

    __device__ auto operator()(FluidParticlesData particles,
                               uint32_t id,
                               const NeighborGrid::Device& grid,
                               const Simulation::Parameters& simulationData) const -> float;

private:
    float _maximalMass;
    InterfaceParameters _interface;
};

inline __device__ auto SplitCriterionGenerator::operator()(FluidParticlesData particles,
                                                           uint32_t id,
                                                           const NeighborGrid::Device& grid,
                                                           const Simulation::Parameters& simulationData) const -> float
{
    if (particles.masses[id] < _minimalMass)
    {
        return -1.0F;
    }

    const auto position = particles.positions[id];
    const auto smoothingRadius = particles.smoothingRadii[id];

    if (!checkIfParticleCanBeSplitInsideDomain(simulationData.domain, position, smoothingRadius, _splitting.epsilon))
    {
        return -1.0F;
    }

    // Compute weighted average displacement vector: u_i = sum(m_j * W * r_ij) / sum(m_j * W)
    auto weightedDisplacement = glm::vec3(0.0F);
    auto weightSum = 0.0F;

    grid.forEachFluidNeighbor(position,
                              particles.positions,
                              smoothingRadius,
                              [&](const auto neighbourIdx, const glm::vec4& adjustedPos) {
                                  if (neighbourIdx == id)
                                  {
                                      return;
                                  }

                                  const auto r_ij = glm::vec3(adjustedPos - position);
                                  const auto dist = glm::length(r_ij);

                                  const auto neighborMass = particles.masses[neighbourIdx];
                                  const auto neighborSmoothingRadius = particles.smoothingRadii[neighbourIdx];
                                  const auto kernelValue = device::wendlandKernel(dist, neighborSmoothingRadius);
                                  const auto weight = neighborMass * kernelValue;

                                  weightedDisplacement += weight * r_ij;
                                  weightSum += weight;
                              });

    if (weightSum < 1e-6F)
    {
        return -1.0F;
    }

    // Normalize to get the interface proximity indicator
    const auto u_i = weightedDisplacement / weightSum;
    const auto proximityMagnitude = glm::length(u_i);

    // Split criterion: ||u_i|| > u^split
    if (proximityMagnitude > _interface.split.proximityMagnitudeThreshold)
    {
        return proximityMagnitude;
    }

    return -1.0F;
}

inline __device__ auto MergeCriterionGenerator::operator()(FluidParticlesData particles,
                                                           uint32_t id,
                                                           const NeighborGrid::Device& grid,
                                                           const Simulation::Parameters& simulationData) const -> float
{
    if (particles.masses[id] > _maximalMass)
    {
        return -1.0F;
    }
    const auto smoothingRadius = particles.smoothingRadii[id];
    if (!checkIfParticleCanBeSplitInsideDomain(simulationData.domain, particles.positions[id], smoothingRadius, 1.F))
    {
        return -1;
    }

    const auto position = particles.positions[id];

    // Compute weighted average displacement vector: u_i = sum(m_j * W * r_ij) / sum(m_j * W)
    auto weightedDisplacement = glm::vec3(0.0F);
    auto weightSum = 0.0F;

    grid.forEachFluidNeighbor(position,
                              particles.positions,
                              smoothingRadius,
                              [&](const auto neighbourIdx, const glm::vec4& adjustedPos) {
                                  if (neighbourIdx == id)
                                  {
                                      return;
                                  }

                                  const auto r_ij = glm::vec3(adjustedPos - position);
                                  const auto dist = glm::length(r_ij);

                                  if (dist > 1e-6F)
                                  {
                                      const auto neighborMass = particles.masses[neighbourIdx];
                                      const auto neighborSmoothingRadius = particles.smoothingRadii[neighbourIdx];
                                      const auto kernelValue = device::wendlandKernel(dist, neighborSmoothingRadius);
                                      const auto weight = neighborMass * kernelValue;

                                      weightedDisplacement += weight * r_ij;
                                      weightSum += weight;
                                  }
                              });

    if (weightSum < 1e-6F)
    {
        return -1.0F;
    }

    // Normalize to get the interface proximity indicator
    const auto u_i = weightedDisplacement / weightSum;
    const auto proximityMagnitude = glm::length(u_i);

    // Merge criterion: ||u_i|| < u^merge
    if (proximityMagnitude < _interface.merge.proximityMagnitudeThreshold)
    {
        return proximityMagnitude / _interface.merge.proximityMagnitudeThreshold;
    }

    return -1.0F;
}

}
