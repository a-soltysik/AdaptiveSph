#include <device_atomic_functions.h>

#include <array>
#include <cfloat>
#include <cmath>
#include <glm/exponential.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/geometric.hpp>
#include <numbers>
#include <utility>

#include "AdaptiveAlgorithm.cuh"
#include "WendlandKernel.cuh"
#include "cuda/refinement/RefinementParameters.cuh"
#include "cuda/simulation/Simulation.cuh"
#include "simulation/SphSimulation.cuh"
#include "simulation/refinement/Common.cuh"
#include "utils/HelperMath.cuh"

namespace sph::cuda::refinement
{
namespace detail
{
constexpr __device__ auto phi = std::numbers::phi_v<float>;
constexpr __device__ auto invnorm = 0.5257311121F;
}

__device__ auto getNewRadius(float mass, float baseMass, float baseRadius) -> float
{
    return baseRadius * std::cbrt(mass / baseMass);
}

constexpr __constant__ __device__ auto icosahedronVertices = std::array {
    float3 {0.F,                            detail::phi* detail::invnorm,  detail::invnorm              },
    float3 {0.F,                            detail::phi* detail::invnorm,  -detail::invnorm             },
    float3 {0.F,                            -detail::phi* detail::invnorm, -detail::invnorm             },
    float3 {0.F,                            -detail::phi* detail::invnorm, detail::invnorm              },
    float3 {detail::invnorm,                0,                             detail::phi* detail::invnorm },
    float3 {-detail::invnorm,               0,                             detail::phi* detail::invnorm },
    float3 {detail::invnorm,                0,                             -detail::phi* detail::invnorm},
    float3 {-detail::invnorm,               0,                             -detail::phi* detail::invnorm},
    float3 {detail::phi * detail::invnorm,  detail::invnorm,               0                            },
    float3 {detail::phi * detail::invnorm,  -detail::invnorm,              0                            },
    float3 {-detail::phi * detail::invnorm, detail::invnorm,               0                            },
    float3 {-detail::phi * detail::invnorm, -detail::invnorm,              0                            }
};

__device__ auto calculateMergedSmoothingLength(std::pair<glm::vec4, glm::vec4> positions,
                                               std::pair<float, float> masses,
                                               std::pair<float, float> smoothingRadiuses,
                                               glm::vec4 mergedPos,
                                               float mergedMass) -> float
{
    const auto distances = std::pair {glm::length(glm::vec3 {positions.first - mergedPos}),
                                      glm::length(glm::vec3 {positions.second - mergedPos})};

    const auto kernelValues = std::pair {device::wendlandKernel(distances.first, smoothingRadiuses.first),
                                         device::wendlandKernel(distances.second, smoothingRadiuses.second)};

    const auto denomNormal = (masses.first * kernelValues.first) + (masses.second * kernelValues.second);

    static constexpr auto coefficient = 16.0F * std::numbers::pi_v<float> / 21.F;

    return std::cbrt(coefficient * mergedMass / denomNormal);
}

__global__ void splitParticles(FluidParticlesData particles,
                               RefinementDataView refinementData,
                               SplittingParameters params,
                               uint32_t maxParticleCount)
{
    const auto tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= *refinementData.split.particlesSplitCount)
    {
        return;
    }

    const auto particleIdx = refinementData.split.particlesIdsToSplit[tid];
    const auto newParticleBase = atomicAdd(refinementData.particlesCount, icosahedronVertices.size());

    if (newParticleBase + icosahedronVertices.size() > maxParticleCount)
    {
        atomicSub(refinementData.particlesCount, icosahedronVertices.size());
        return;
    }
    const auto originalPosition = particles.positions[particleIdx];
    const auto originalVelocity = particles.velocities[particleIdx];
    const auto originalAcceleration = particles.accelerations[particleIdx];
    const auto originalMass = particles.masses[particleIdx];
    const auto originalRadius = particles.radii[particleIdx];
    const auto originalSmoothingLength = particles.smoothingRadii[particleIdx];
    const float daughterMass = originalMass * params.vertexMassRatio;
    const float centerMass = originalMass * params.centerMassRatio;
    const float newRadius = getNewRadius(daughterMass, originalMass, originalRadius);
    const float newSmoothingLength = params.alpha * originalSmoothingLength;

    particles.masses[particleIdx] = centerMass;
    particles.radii[particleIdx] = getNewRadius(centerMass, originalMass, originalRadius);
    particles.smoothingRadii[particleIdx] = newSmoothingLength;

    for (uint32_t i = 0; i < icosahedronVertices.size(); i++)
    {
        const auto newIdx = newParticleBase + i;

        const auto offset = icosahedronVertices[i] * params.epsilon * originalSmoothingLength;
        particles.positions[newIdx] = originalPosition + glm::vec4(offset.x, offset.y, offset.z, 0.0F);
        particles.velocities[newIdx] = originalVelocity;
        particles.accelerations[newIdx] = originalAcceleration;
        particles.masses[newIdx] = daughterMass;
        particles.radii[newIdx] = newRadius;
        particles.smoothingRadii[newIdx] = newSmoothingLength;

        particles.densities[newIdx] = 0.0F;
    }
}

__device__ void updateParticleCount(RefinementDataView refinementData, uint32_t particleCount)
{
    auto removedCount = refinementData.merge.prefixSums[particleCount - 1];
    if (refinementData.merge.removalFlags[particleCount - 1] == RefinementData::RemovalState::Remove)
    {
        removedCount++;
    }

    *refinementData.particlesCount = particleCount - removedCount;
}

__global__ void compactParticles(FluidParticlesData particles,
                                 RefinementDataView refinementData,
                                 uint32_t particleCount)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particleCount)
    {
        return;
    }

    if (refinementData.merge.removalFlags[idx] == RefinementData::RemovalState::Keep)
    {
        const auto newIdx = idx - refinementData.merge.prefixSums[idx];

        particles.positions[newIdx] = particles.positions[idx];
        particles.velocities[newIdx] = particles.velocities[idx];
        particles.accelerations[newIdx] = particles.accelerations[idx];
        particles.masses[newIdx] = particles.masses[idx];
        particles.radii[newIdx] = particles.radii[idx];
        particles.smoothingRadii[newIdx] = particles.smoothingRadii[idx];
        particles.densities[newIdx] = particles.densities[idx];
    }

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        updateParticleCount(refinementData, particleCount);
    }
}

__global__ void identifyMergeCandidates(FluidParticlesData particles,
                                        RefinementDataView refinementData,
                                        NeighborGrid::Device grid,
                                        Simulation::Parameters simulationData,
                                        RefinementParameters refinementParameters)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *refinementData.merge.eligibleCount)
    {
        return;
    }
    const auto particleId = refinementData.merge.eligibleParticles[idx];
    const auto position = particles.positions[particleId];
    const auto smoothingRadius = particles.smoothingRadii[particleId];
    auto closestDistanceSquared = FLT_MAX;
    auto closestIdx = constants::noCandidate;

    grid.forEachFluidNeighbor(
        position,
        particles.positions,
        smoothingRadius,
        [&](const auto neighborIdx, const glm::vec4& neighborPosition) {
            if (neighborIdx == particleId)
            {
                return;
            }
            if (refinementData.merge.removalFlags[neighborIdx] != RefinementData::RemovalState::Keep)
            {
                return;
            }

            const auto offset = neighborPosition - position;
            const auto distanceSquared = glm::dot(offset, offset);
            const auto neighbourSmoothingRadius = particles.smoothingRadii[neighborIdx];
            const auto minimalSmoothingRadius = glm::min(smoothingRadius, neighbourSmoothingRadius);
            const auto neighbourMass = particles.masses[neighborIdx];

            if (distanceSquared < minimalSmoothingRadius * minimalSmoothingRadius &&
                distanceSquared < closestDistanceSquared &&
                neighbourMass < refinementParameters.maxMassRatio * simulationData.baseParticleMass)
            {
                closestDistanceSquared = distanceSquared;
                closestIdx = neighborIdx;
            }
        });

    refinementData.merge.mergeCandidates[particleId] = closestIdx;
}

__global__ void resolveMergePairs(RefinementDataView refinementData, uint32_t particleCount)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particleCount)
    {
        return;
    }

    if (refinementData.merge.mergeCandidates[idx] == constants::noCandidate ||
        refinementData.merge.removalFlags[idx] != RefinementData::RemovalState::Keep)
    {
        return;
    }
    const auto candidateIdx = refinementData.merge.mergeCandidates[idx];

    if (refinementData.merge.mergeCandidates[candidateIdx] == idx)
    {
        if (idx < candidateIdx)
        {
            const auto pairIdx = atomicAdd(refinementData.merge.mergeCount, 1);
            refinementData.merge.mergePairs[2 * pairIdx] = idx;
            refinementData.merge.mergePairs[2 * pairIdx + 1] = candidateIdx;

            refinementData.merge.removalFlags[candidateIdx] = RefinementData::RemovalState::Remove;
        }
    }
}

__global__ void performMerges(FluidParticlesData particles,
                              RefinementDataView refinementData,
                              Simulation::Parameters simulationData)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *refinementData.merge.mergeCount)
    {
        return;
    }

    const auto keepIdx = refinementData.merge.mergePairs[2 * idx];
    const auto removeIdx = refinementData.merge.mergePairs[2 * idx + 1];

    if (keepIdx == constants::noCandidate || removeIdx == constants::noCandidate || keepIdx == removeIdx ||
        keepIdx >= particles.particleCount || removeIdx >= particles.particleCount)
    {
        return;
    }

    const auto positions = std::pair {particles.positions[keepIdx], particles.positions[removeIdx]};
    const auto masses = std::pair {particles.masses[keepIdx], particles.masses[removeIdx]};
    const auto smoothingRadii = std::pair {particles.smoothingRadii[keepIdx], particles.smoothingRadii[removeIdx]};
    const auto velocities = std::pair {particles.velocities[keepIdx], particles.velocities[removeIdx]};
    const auto accelerations = std::pair {particles.accelerations[keepIdx], particles.accelerations[removeIdx]};

    const auto newMass = masses.first + masses.second;
    const auto newPosition = (masses.first * positions.first + masses.second * positions.second) / newMass;
    const auto newVelocity = (masses.first * velocities.first + masses.second * velocities.second) / newMass;
    const auto newAcceleration = (masses.first * accelerations.first + masses.second * accelerations.second) / newMass;

    particles.positions[keepIdx] = newPosition;
    particles.velocities[keepIdx] = newVelocity;
    particles.accelerations[keepIdx] = newAcceleration;
    particles.masses[keepIdx] = newMass;
    particles.radii[keepIdx] = simulationData.baseParticleRadius * std::cbrt(newMass / simulationData.baseParticleMass);
    particles.smoothingRadii[keepIdx] =
        calculateMergedSmoothingLength(positions, masses, smoothingRadii, newPosition, newMass);

    particles.densities[keepIdx] = 0.0F;
}

}
