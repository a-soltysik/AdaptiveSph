#include <device_atomic_functions.h>

#include <array>
#include <cfloat>
#include <climits>
#include <cmath>
#include <glm/exponential.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/geometric.hpp>
#include <numbers>
#include <utility>

#include "AdaptiveAlgorithm.cuh"
#include "algorithm/kernels/Kernel.cuh"
#include "cuda/Simulation.cuh"
#include "cuda/refinement/RefinementParameters.cuh"
#include "glm/ext/scalar_constants.hpp"
#include "simulation/adaptive/SphSimulation.cuh"
#include "utils/Iteration.cuh"

namespace sph::cuda::refinement
{
namespace detail
{
__device__ __constant__ float phi = std::numbers::phi_v<float>;
__device__ __constant__ float invnorm = 0.5257311121F;
}

__device__ auto getNewRadius(float mass, float baseMass, float baseRadius) -> float
{
    return baseRadius * glm::pow(mass / baseMass, 1.0F / 3.0F);
}

__device__ auto getIcosahedronVertices() -> std::array<glm::vec3, 12>
{
    return std::array {
        glm::vec3 {0.F,                            detail::phi * detail::invnorm,  detail::invnorm               },
        glm::vec3 {0.F,                            detail::phi * detail::invnorm,  -detail::invnorm              },
        glm::vec3 {0.F,                            -detail::phi * detail::invnorm, -detail::invnorm              },
        glm::vec3 {0.F,                            -detail::phi * detail::invnorm, detail::invnorm               },
        glm::vec3 {detail::invnorm,                0,                              detail::phi * detail::invnorm },
        glm::vec3 {-detail::invnorm,               0,                              detail::phi * detail::invnorm },
        glm::vec3 {detail::invnorm,                0,                              -detail::phi * detail::invnorm},
        glm::vec3 {-detail::invnorm,               0,                              -detail::phi * detail::invnorm},
        glm::vec3 {detail::phi * detail::invnorm,  detail::invnorm,                0                             },
        glm::vec3 {detail::phi * detail::invnorm,  -detail::invnorm,               0                             },
        glm::vec3 {-detail::phi * detail::invnorm, detail::invnorm,                0                             },
        glm::vec3 {-detail::phi * detail::invnorm, -detail::invnorm,               0                             }
    };
}

__device__ auto calculateMergedSmoothingLength(std::pair<glm::vec4, glm::vec4> positions,
                                               std::pair<float, float> masses,
                                               std::pair<float, float> smoothingRadiuses,
                                               glm::vec4 mergedPos,
                                               float mergedMass,
                                               float weightNormal = 0.7F,
                                               float weightNear = 0.3F) -> float
{
    const auto distances = std::pair {glm::length(glm::vec3 {positions.first - mergedPos}),
                                      glm::length(glm::vec3 {positions.second - mergedPos})};

    const auto kernelValues = std::pair {device::densityKernel(distances.first, smoothingRadiuses.first),
                                         device::densityKernel(distances.second, smoothingRadiuses.second)};

    const auto denomNormal = (masses.first * kernelValues.first) + (masses.second * kernelValues.second);

    const auto nearKernelValues = std::pair {device::nearDensityKernel(distances.first, smoothingRadiuses.first),
                                             device::nearDensityKernel(distances.second, smoothingRadiuses.second)};
    const auto denomNear = (masses.first * nearKernelValues.first) + (masses.second * nearKernelValues.second);

    const auto normalSmoothingRadius = std::cbrt((16.0F * glm::pi<float>() * mergedMass) / (21.F * denomNormal));
    const auto nearSmoothingRadius = std::cbrt((15.0F * mergedMass) / (glm::pi<float>() * denomNear));

    return ((weightNormal * normalSmoothingRadius) + (weightNear * nearSmoothingRadius));
}

__global__ void splitParticles(ParticlesData particles,
                               RefinementData refinementData,
                               SplittingParameters params,
                               uint32_t maxParticleCount)
{
    const auto tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= *refinementData.split.particlesSplitCount)
    {
        return;
    }
    const auto particleIdx = refinementData.split.particlesIdsToSplit[tid];
    const auto icosahedronVertices = getIcosahedronVertices();
    const auto newParticleBase = atomicAdd(refinementData.particlesCount, icosahedronVertices.size());

    if (newParticleBase + icosahedronVertices.size() > maxParticleCount)
    {
        atomicSub(refinementData.particlesCount, icosahedronVertices.size());
        return;
    }
    const auto originalPosition = particles.positions[particleIdx];
    const auto originalVelocity = particles.velocities[particleIdx];
    const auto originalMass = particles.masses[particleIdx];
    const auto originalRadius = particles.radiuses[particleIdx];
    const auto originalSmoothingLength = particles.smoothingRadiuses[particleIdx];
    const float daughterMass = originalMass * params.vertexMassRatio;
    const float centerMass = originalMass * params.centerMassRatio;
    const float newRadius = getNewRadius(daughterMass, originalMass, originalRadius);
    const float newSmoothingLength = params.alpha * originalSmoothingLength;

    particles.masses[particleIdx] = centerMass;
    particles.radiuses[particleIdx] = getNewRadius(centerMass, originalMass, originalRadius);
    particles.smoothingRadiuses[particleIdx] = newSmoothingLength;

    for (uint32_t i = 0; i < icosahedronVertices.size(); i++)
    {
        const auto newIdx = newParticleBase + i;

        const auto offset = icosahedronVertices[i] * params.epsilon * originalSmoothingLength;
        particles.positions[newIdx] = originalPosition + glm::vec4(offset, 0.0F);
        particles.predictedPositions[newIdx] = particles.positions[newIdx];
        particles.velocities[newIdx] = originalVelocity;
        particles.masses[newIdx] = daughterMass;
        particles.radiuses[newIdx] = newRadius;
        particles.smoothingRadiuses[newIdx] = newSmoothingLength;

        particles.densities[newIdx] = 0.0F;
        particles.nearDensities[newIdx] = 0.0F;
        particles.pressures[newIdx] = 0.0F;
    }
}

__global__ void updateParticleCount(RefinementData refinementData, uint32_t particleCount)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        auto removedCount = refinementData.merge.prefixSums[particleCount - 1];
        if (refinementData.merge.removalFlags[particleCount - 1] == RefinementData::RemovalState::Remove)
        {
            removedCount++;
        }

        *refinementData.particlesCount = particleCount - removedCount;
    }
}

__global__ void compactParticles(ParticlesData particles, RefinementData::MergeData mergeData, uint32_t particleCount)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particleCount)
    {
        return;
    }

    if (mergeData.removalFlags[idx] == RefinementData::RemovalState::Keep)
    {
        const auto newIdx = idx - mergeData.prefixSums[idx];

        particles.positions[newIdx] = particles.positions[idx];
        particles.predictedPositions[newIdx] = particles.predictedPositions[idx];
        particles.velocities[newIdx] = particles.velocities[idx];
        particles.masses[newIdx] = particles.masses[idx];
        particles.radiuses[newIdx] = particles.radiuses[idx];
        particles.smoothingRadiuses[newIdx] = particles.smoothingRadiuses[idx];
        particles.densities[newIdx] = particles.densities[idx];
        particles.nearDensities[newIdx] = particles.nearDensities[idx];
        particles.pressures[newIdx] = particles.pressures[idx];
    }
}

__global__ void identifyMergeCandidates(ParticlesData particles,
                                        RefinementData::MergeData mergeData,
                                        SphSimulation::Grid grid,
                                        Simulation::Parameters simulationData,
                                        RefinementParameters refinementParameters)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *mergeData.eligibleCount)
    {
        return;
    }
    const auto particleId = mergeData.eligibleParticles[idx];

    mergeData.mergeCandidates[particleId] = UINT_MAX;

    const auto position = particles.positions[particleId];
    const auto smoothingRadius = particles.smoothingRadiuses[particleId];
    auto closestDistanceSquared = FLT_MAX;
    auto closestIdx = UINT_MAX;

    forEachNeighbour(position,
                     particles,
                     simulationData,
                     grid,
                     [&](const auto neighborIdx, const glm::vec4& adjustedPos) {
                         if (neighborIdx == particleId)
                         {
                             return;
                         }
                         if (mergeData.removalFlags[neighborIdx] != RefinementData::RemovalState::Keep)
                         {
                             return;
                         }

                         const auto offset = adjustedPos - position;
                         const auto distanceSquared = glm::dot(offset, offset);
                         const auto neighbourSmoothingRadius = particles.smoothingRadiuses[neighborIdx];
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
    if (closestIdx != UINT_MAX)
    {
        mergeData.mergeCandidates[particleId] = closestIdx;
    }
}

__global__ void resolveMergePairs(RefinementData::MergeData mergeData, uint32_t particleCount)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particleCount)
    {
        return;
    }

    if (mergeData.mergeCandidates[idx] == UINT_MAX || mergeData.removalFlags[idx] != RefinementData::RemovalState::Keep)
    {
        return;
    }
    const auto candidateIdx = mergeData.mergeCandidates[idx];

    if (mergeData.mergeCandidates[candidateIdx] == idx)
    {
        // Only the smaller index should process the merge to avoid duplications
        if (idx < candidateIdx)
        {
            const auto pairIdx = atomicAdd(mergeData.mergeCount, 1);
            // Store the pair (smaller index first)
            mergeData.mergePairs[2 * pairIdx] = idx;
            mergeData.mergePairs[2 * pairIdx + 1] = candidateIdx;

            mergeData.removalFlags[candidateIdx] = RefinementData::RemovalState::Remove;
        }
    }
}

__global__ void performMerges(ParticlesData particles,
                              RefinementData::MergeData mergeData,
                              Simulation::Parameters simulationData)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *mergeData.mergeCount)
    {
        return;
    }

    const auto keepIdx = mergeData.mergePairs[2 * idx];
    const auto removeIdx = mergeData.mergePairs[2 * idx + 1];

    if (keepIdx == UINT_MAX || removeIdx == UINT_MAX || keepIdx == removeIdx || keepIdx >= particles.particleCount ||
        removeIdx >= particles.particleCount)
    {
        return;
    }

    const auto positions = std::pair {particles.positions[keepIdx], particles.positions[removeIdx]};
    const auto masses = std::pair {particles.masses[keepIdx], particles.masses[removeIdx]};
    const auto smoothingRadiuses =
        std::pair {particles.smoothingRadiuses[keepIdx], particles.smoothingRadiuses[removeIdx]};
    const auto velocities = std::pair {particles.velocities[keepIdx], particles.velocities[removeIdx]};

    const auto newMass = masses.first + masses.second;
    const auto newPos = (masses.first * positions.first + masses.second * positions.second) / newMass;
    const auto newVel = (masses.first * velocities.first + masses.second * velocities.second) / newMass;

    particles.positions[keepIdx] = newPos;
    particles.predictedPositions[keepIdx] = newPos;
    particles.velocities[keepIdx] = newVel;
    particles.masses[keepIdx] = newMass;
    particles.radiuses[keepIdx] =
        simulationData.baseParticleRadius * std::cbrt(newMass / simulationData.baseParticleMass);
    particles.smoothingRadiuses[keepIdx] =
        calculateMergedSmoothingLength(positions, masses, smoothingRadiuses, newPos, newMass);

    particles.densities[keepIdx] = 0.0F;
    particles.nearDensities[keepIdx] = 0.0F;
    particles.pressures[keepIdx] = 0.0F;
}

}
