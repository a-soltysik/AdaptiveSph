#include <device_atomic_functions.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

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

struct IsEligibleMergeCriterion
{
    __host__ __device__ auto operator()(float value) const -> bool
    {
        return value >= 0.0F;
    }
};

struct MergeCriterionLess
{
    const float* criterionValues;

    __host__ __device__ auto operator()(uint32_t left, uint32_t right) const -> bool
    {
        const auto leftValue = criterionValues[left];
        const auto rightValue = criterionValues[right];
        return leftValue < rightValue || (leftValue == rightValue && left < right);
    }
};

struct SelectedMaskValue
{
    __host__ __device__ auto operator()(uint32_t) const -> uint32_t
    {
        return 1U;
    }
};

struct IsMarkedForRemoval
{
    __host__ __device__ auto operator()(RefinementData::RemovalState state) const -> bool
    {
        return state == RefinementData::RemovalState::Remove;
    }
};
}

auto findTopParticlesToMerge(FluidParticlesData particles,
                             RefinementData& refinementData,
                             RefinementParameters refinementParameters) -> void
{
    cudaDeviceSynchronize();

    const auto particleIdsBegin = thrust::make_counting_iterator<uint32_t>(0);
    const auto particleIdsEnd = particleIdsBegin + particles.particleCount;
    const auto criterionValues = refinementData.merge.criterionValues.data();
    const auto eligibleParticlesBegin = refinementData.merge.eligibleParticles.begin();
    const auto eligibleParticlesEnd = thrust::copy_if(thrust::device,
                                                      particleIdsBegin,
                                                      particleIdsEnd,
                                                      refinementData.merge.criterionValues.begin(),
                                                      eligibleParticlesBegin,
                                                      detail::IsEligibleMergeCriterion {});
    const auto qualifiedCount = static_cast<uint32_t>(eligibleParticlesEnd - eligibleParticlesBegin);

    const auto maxEligibleCount =
        std::min(qualifiedCount,
                 static_cast<uint32_t>(
                     std::round(static_cast<float>(particles.particleCount) * refinementParameters.maxBatchRatio)));

    // Reuse the split ID scratch buffer as a membership mask during merge selection.
    thrust::fill(thrust::device,
                 refinementData.particlesIds.begin(),
                 refinementData.particlesIds.begin() + particles.particleCount,
                 0U);

    if (maxEligibleCount > 0)
    {
        thrust::sort(thrust::device,
                     eligibleParticlesBegin,
                     eligibleParticlesEnd,
                     detail::MergeCriterionLess {thrust::raw_pointer_cast(criterionValues)});

        const auto selectedMaskValuesBegin =
            thrust::make_transform_iterator(eligibleParticlesBegin, detail::SelectedMaskValue {});
        thrust::scatter(thrust::device,
                        selectedMaskValuesBegin,
                        selectedMaskValuesBegin + maxEligibleCount,
                        eligibleParticlesBegin,
                        refinementData.particlesIds.begin());
    }

    refinementData.merge.eligibleCount = maxEligibleCount;
}

auto compactParticles(FluidParticlesData particles, RefinementData& refinementData) -> uint32_t
{
    if (particles.particleCount == 0)
    {
        return 0;
    }

    auto particlesBegin = thrust::make_zip_iterator(thrust::make_tuple(particles.positions,
                                                                       particles.velocities,
                                                                       particles.accelerations,
                                                                       particles.densities,
                                                                       particles.radii,
                                                                       particles.smoothingRadii,
                                                                       particles.masses));
    const auto particlesEnd = particlesBegin + particles.particleCount;
    const auto newEnd = thrust::remove_if(thrust::device,
                                          particlesBegin,
                                          particlesEnd,
                                          refinementData.merge.removalFlags.begin(),
                                          detail::IsMarkedForRemoval {});
    return static_cast<uint32_t>(newEnd - particlesBegin);
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
    const auto maximalMergeMass = refinementParameters.maxMassRatio * simulationData.baseParticleMass;
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
            if (refinementData.merge.selectedMask[neighborIdx] == 0)
            {
                return;
            }

            const auto offset = neighborPosition - position;
            const auto distanceSquared = glm::dot(offset, offset);
            const auto neighbourSmoothingRadius = particles.smoothingRadii[neighborIdx];
            const auto minimalSmoothingRadius = glm::min(smoothingRadius, neighbourSmoothingRadius);
            const auto mergedMass = particles.masses[particleId] + particles.masses[neighborIdx];

            if (distanceSquared < minimalSmoothingRadius * minimalSmoothingRadius &&
                distanceSquared < closestDistanceSquared && mergedMass <= maximalMergeMass)
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

    const auto candidateIdx = refinementData.merge.mergeCandidates[idx];
    if (candidateIdx == constants::noCandidate || candidateIdx >= particleCount || candidateIdx == idx)
    {
        return;
    }

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
    const auto newSmoothingRadius =
        calculateMergedSmoothingLength(positions, masses, smoothingRadii, newPosition, newMass);
    const auto newRadius = simulationData.baseParticleRadius * std::cbrt(newMass / simulationData.baseParticleMass);

    particles.positions[keepIdx] = newPosition;
    particles.velocities[keepIdx] = newVelocity;
    particles.accelerations[keepIdx] = newAcceleration;
    particles.masses[keepIdx] = newMass;
    particles.radii[keepIdx] = newRadius;
    particles.smoothingRadii[keepIdx] = newSmoothingRadius;

    particles.densities[keepIdx] = 0.0F;
}

}
