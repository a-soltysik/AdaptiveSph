#include <device_atomic_functions.h>

#include <array>
#include <glm/exponential.hpp>
#include <glm/geometric.hpp>

#include "../common/Iteration.hpp"
#include "../common/Utils.cuh"
#include "../device/Kernel.cuh"
#include "ParticleOperations.cuh"
#include "cuda/Simulation.cuh"
#include "glm/ext/scalar_constants.hpp"

namespace sph::cuda::refinement
{
namespace detail
{
__device__ __constant__ float phi = 1.61803398875f;
__device__ __constant__ float invnorm = 0.5257311121f;
}

__device__ float getNewRadius(float mass, float baseMass, float baseRadius)
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

__device__ float calculateMergedSmoothingLength(const glm::vec4& posA,
                                                float massA,
                                                float hA,
                                                const glm::vec4& posB,
                                                float massB,
                                                float hB,
                                                const glm::vec4& mergedPos,
                                                float mergedMass,
                                                float weightNormal = 0.9f,
                                                float weightNear = .1f)
{
    const float distA = glm::length(glm::vec3(posA - mergedPos));
    const float distB = glm::length(glm::vec3(posB - mergedPos));

    const float W_MA = device::densityKernel(distA, hA);
    const float W_MB = device::densityKernel(distB, hB);

    const float denomNormal = massA * W_MA + massB * W_MB;
    const float W_MA_near = device::nearDensityKernel(distA, hA);
    const float W_MB_near = device::nearDensityKernel(distB, hB);
    const float denomNear = massA * W_MA_near + massB * W_MB_near;

    const float h_normal = cbrtf((16.0f * glm::pi<float>() * mergedMass) / (21.f * denomNormal));
    const float h_near = cbrtf((15.0f * mergedMass) / (glm::pi<float>() * denomNear));

    return (weightNormal * h_normal + weightNear * h_near) / 2.F;  //is better with 2
}

__global__ void splitParticles(ParticlesData particles,
                               RefinementData refinementData,
                               SplittingParameters params,
                               uint32_t maxParticleCount)
{
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= *refinementData.split.particlesSplitCount)
    {
        return;
    }
    const auto particleIdx = refinementData.split.particlesIdsToSplit.data[tid];
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

__global__ void mergeParticles(ParticlesData particles,
                               RefinementData refinementData,
                               Simulation::Parameters simulationData)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *refinementData.merge.particlesMergeCount)
    {
        return;
    }
    const auto keepIdx = refinementData.merge.particlesIdsToMerge.first.data[idx];
    const auto removeIdx = refinementData.merge.particlesIdsToMerge.second.data[idx];
    if (keepIdx == UINT_MAX || removeIdx == UINT_MAX)
    {
        return;
    }
    if (keepIdx == removeIdx || keepIdx >= particles.particleCount || removeIdx >= particles.particleCount)
    {
        return;
    }

    if (refinementData.merge.removalFlags.data[keepIdx] != RefinementData::RemovalState::Keep ||
        refinementData.merge.removalFlags.data[removeIdx] != RefinementData::RemovalState::Remove)
    {
        return;
    }

    const float removeMass = particles.masses[removeIdx];
    const float addedMass = atomicAdd(&particles.masses[keepIdx], removeMass);
    if (addedMass == particles.masses[keepIdx] - removeMass)
    {
        const auto positions = std::pair {particles.positions[keepIdx], particles.positions[removeIdx]};
        const auto velocities = std::pair {particles.velocities[keepIdx], particles.velocities[removeIdx]};
        const auto smoothingRadiuses =
            std::pair {particles.smoothingRadiuses[keepIdx], particles.smoothingRadiuses[removeIdx]};
        const auto masses = std::pair {addedMass, removeMass};
        const float newMass = addedMass + removeMass;

        const float newRadius =
            simulationData.baseParticleRadius * powf(newMass / simulationData.baseParticleMass, 1.0f / 3.0f);
        const auto newPosition = (masses.first * positions.first + masses.second * positions.second) / newMass;
        const auto newVelocity = (masses.first * velocities.first + masses.second * velocities.second) / newMass;

        particles.positions[keepIdx] = newPosition;
        particles.predictedPositions[keepIdx] = newPosition;
        particles.velocities[keepIdx] = newVelocity;
        particles.radiuses[keepIdx] = newRadius;

        particles.smoothingRadiuses[keepIdx] = calculateMergedSmoothingLength(positions.first,
                                                                              masses.first,
                                                                              smoothingRadiuses.first,
                                                                              positions.second,
                                                                              masses.second,
                                                                              smoothingRadiuses.second,
                                                                              newPosition,
                                                                              newMass);

        particles.densities[keepIdx] = 0.0f;
        particles.nearDensities[keepIdx] = 0.0f;
        particles.pressures[keepIdx] = 0.0f;
    }
}

__global__ void validateMergePairs(RefinementData refinementData, uint32_t particleCount)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *refinementData.merge.particlesMergeCount)
    {
        return;
    }

    const auto keepIdx = refinementData.merge.particlesIdsToMerge.first.data[idx];
    const auto removeIdx = refinementData.merge.particlesIdsToMerge.second.data[idx];

    if (keepIdx == UINT_MAX || removeIdx == UINT_MAX || keepIdx >= particleCount || removeIdx >= particleCount ||
        refinementData.merge.removalFlags.data[keepIdx] != RefinementData::RemovalState::Keep ||
        refinementData.merge.removalFlags.data[removeIdx] != RefinementData::RemovalState::Remove)
    {
        refinementData.merge.particlesIdsToMerge.first.data[idx] = UINT_MAX;
        refinementData.merge.particlesIdsToMerge.second.data[idx] = UINT_MAX;
    }
}

__device__ auto findClosestParticle(const ParticlesData& particles,
                                    uint32_t particleIdx,
                                    const SphSimulation::Grid& grid,
                                    const Simulation::Parameters& simulationData) -> std::pair<uint32_t, float>
{
    const auto position = particles.positions[particleIdx];
    const auto originCell = calculateCellIndex(position, simulationData, grid);

    auto result = std::pair {particleIdx, FLT_MAX};

    for (const auto offset : offsets)
    {
        const auto range = getStartEndIndices(originCell + glm::uvec3 {offset.x, offset.y, offset.z}, grid);

        if (range.first == -1 || range.first > range.second)
        {
            continue;
        }

        for (auto i = range.first; i <= range.second; i++)
        {
            const auto neighborIdx = grid.particleArrayIndices.data[i];

            if (neighborIdx == particleIdx)
            {
                continue;
            }

            const auto neighborPos = particles.positions[neighborIdx];
            const auto offsetVec = neighborPos - position;
            const auto distSq = glm::dot(offsetVec, offsetVec);

            if (distSq < result.second &&
                distSq < particles.smoothingRadiuses[particleIdx] * particles.smoothingRadiuses[particleIdx])
            {
                result.second = distSq;
                result.first = neighborIdx;
            }
        }
    }

    return {result.first, glm::sqrt(result.second)};
}

__global__ void removeParticles(ParticlesData particles, RefinementData refinementData)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }

    if (refinementData.merge.removalFlags.data[idx] != RefinementData::RemovalState::Remove)
    {
        const auto newId = idx - refinementData.merge.prefixSums.data[idx];
        if (newId != idx)
        {
            particles.positions[newId] = particles.positions[idx];
            particles.predictedPositions[newId] = particles.predictedPositions[idx];
            particles.velocities[newId] = particles.velocities[idx];
            particles.masses[newId] = particles.masses[idx];
            particles.radiuses[newId] = particles.radiuses[idx];
            particles.smoothingRadiuses[newId] = particles.smoothingRadiuses[idx];
            particles.densities[newId] = particles.densities[idx];
            particles.nearDensities[newId] = particles.nearDensities[idx];
            particles.pressures[newId] = particles.pressures[idx];
        }
    }
}

__global__ void getMergeCandidates(ParticlesData particles,
                                   RefinementData refinementData,
                                   SphSimulation::Grid grid,
                                   Simulation::Parameters simulationData)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *refinementData.merge.particlesMergeCount)
    {
        return;
    }
    refinementData.merge.particlesIdsToMerge.second.data[idx] = UINT_MAX;
    const auto particleId = refinementData.merge.particlesIdsToMerge.first.data[idx];
    if (particleId >= particles.particleCount)
    {
        return;
    }

    auto closestResult = findClosestParticle(particles, particleId, grid, simulationData);
    if (closestResult.first != particleId && closestResult.first < particles.particleCount &&
        closestResult.second < particles.smoothingRadiuses[particleId])
    {
        refinementData.merge.particlesIdsToMerge.second.data[idx] = closestResult.first;
    }
}

__global__ void updateParticleCount(RefinementData refinementData, uint32_t particleCount)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        uint32_t removedCount = refinementData.merge.prefixSums.data[particleCount - 1];
        if (refinementData.merge.removalFlags.data[particleCount - 1] == RefinementData::RemovalState::Remove)
        {
            removedCount++;
        }

        *refinementData.particlesCount = particleCount - removedCount;
    }
}

__global__ void markPotentialMerges(ParticlesData particles, RefinementData refinementData)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *refinementData.merge.particlesMergeCount)
    {
        return;
    }
    const auto first = refinementData.merge.particlesIdsToMerge.first.data[idx];
    const auto second = refinementData.merge.particlesIdsToMerge.second.data[idx];
    refinementData.merge.particlesIdsToMerge.first.data[idx] = UINT_MAX;
    refinementData.merge.particlesIdsToMerge.second.data[idx] = UINT_MAX;

    if (second == UINT_MAX || first == second || first >= particles.particleCount || second >= particles.particleCount)
    {
        return;
    }

    const auto keepIdx = min(first, second);
    const auto removeIdx = max(first, second);
    const auto keepResult = atomicCAS(reinterpret_cast<int*>(&refinementData.merge.removalFlags.data[keepIdx]),
                                      static_cast<int>(RefinementData::RemovalState::Default),
                                      static_cast<int>(RefinementData::RemovalState::Keep));
    if (keepResult != static_cast<int>(RefinementData::RemovalState::Default))
    {
        return;
    }
    const auto removeResult = atomicCAS(reinterpret_cast<int*>(&refinementData.merge.removalFlags.data[removeIdx]),
                                        static_cast<int>(RefinementData::RemovalState::Default),
                                        static_cast<int>(RefinementData::RemovalState::Remove));

    if (removeResult != static_cast<int>(RefinementData::RemovalState::Default))
    {
        atomicExch(reinterpret_cast<int*>(&refinementData.merge.removalFlags.data[keepIdx]),
                   static_cast<int>(RefinementData::RemovalState::Default));
        return;
    }

    refinementData.merge.particlesIdsToMerge.first.data[idx] = keepIdx;
    refinementData.merge.particlesIdsToMerge.second.data[idx] = removeIdx;
}

__device__ float computeMergeScore(float distance, float neighborMass, float neighborCriterion)
{
    const float distanceWeight = 0.8f;
    const float massWeight = 0.1f;
    const float criterionWeight = 0.1f;
    const float normalizedDistance = distance / 1.0f;
    float normalizedMass = neighborMass / 10.0f;
    const float normalizedCriterion = 1.0f - neighborCriterion;
    return distanceWeight * normalizedDistance + massWeight * normalizedMass + criterionWeight * normalizedCriterion;
}

__global__ void proposePartners(ParticlesData particles,
                                EnhancedMergeData mergeData,
                                SphSimulation::Grid grid,
                                Simulation::Parameters simulationData,
                                MergeConfiguration mergeConfig)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *mergeData.eligibleCount)
    {
        return;
    }

    const auto particleId = mergeData.eligibleParticles.data[idx];
    const auto position = particles.positions[particleId];
    auto bestCandidate = UINT_MAX;
    auto bestScore = FLT_MAX;

    forEachNeighbour(position, simulationData, grid, [&](const auto neighborIdx) {
        if (neighborIdx == particleId)
        {
            return;
        }

        if (mergeData.states.data[neighborIdx].status != MergeState::Status::Available)
        {
            return;
        }

        if (particles.masses[neighborIdx] > mergeConfig.maxMassThreshold)
        {
            return;
        }

        const auto neighborPos = particles.positions[neighborIdx];
        const auto dist = glm::length(glm::vec3(neighborPos - position));
        if (dist < particles.smoothingRadiuses[particleId])
        {
            const auto score =
                computeMergeScore(dist, particles.masses[neighborIdx], mergeData.criterionValues.data[neighborIdx]);
            if (score < bestScore)
            {
                bestScore = score;
                bestCandidate = neighborIdx;
            }
        }
    });

    if (bestCandidate != UINT_MAX)
    {
        auto expected = MergeState::Status::Available;
        if (atomicCAS(reinterpret_cast<uint32_t*>(&mergeData.states.data[particleId].status),
                      static_cast<uint32_t>(expected),
                      static_cast<uint32_t>(MergeState::Status::Proposing)) == static_cast<uint32_t>(expected))
        {
            mergeData.states.data[particleId].partner = bestCandidate;
            mergeData.states.data[particleId].distance = bestScore;
        }
    }
}

__global__ void resolveProposals(ParticlesData particles, EnhancedMergeData mergeData)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }

    if (mergeData.states.data[idx].status != MergeState::Status::Proposing)
    {
        return;
    }

    const auto partnerId = mergeData.states.data[idx].partner;

    bool isMutualProposal = false;
    bool isLowerIndex = false;
    if (partnerId < particles.particleCount && mergeData.states.data[partnerId].status == MergeState::Status::Proposing)
    {
        uint32_t partnerPartner = atomicAdd(&mergeData.states.data[partnerId].partner, 0);
        if (partnerPartner == idx)
        {
            isMutualProposal = true;
            isLowerIndex = (idx < partnerId);
        }
    }

    if (isMutualProposal && isLowerIndex)
    {
        auto expected = MergeState::Status::Proposing;
        auto desired = MergeState::Status::Accepted;

        if (atomicCAS(reinterpret_cast<uint32_t*>(&mergeData.states.data[idx].status),
                      static_cast<uint32_t>(expected),
                      static_cast<uint32_t>(desired)) == static_cast<uint32_t>(expected))
        {
            atomicExch(reinterpret_cast<uint32_t*>(&mergeData.states.data[partnerId].status),
                       static_cast<uint32_t>(MergeState::Status::Paired));
        }
    }
    else if (!isMutualProposal)
    {
        atomicExch(reinterpret_cast<uint32_t*>(&mergeData.states.data[idx].status),
                   static_cast<uint32_t>(MergeState::Status::Available));
    }
}

__global__ void identifyEligibleParticles(ParticlesData particles, EnhancedMergeData mergeData, float maxMass)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }
    mergeData.states.data[idx] = {MergeState::Status::Available, UINT_MAX, FLT_MAX};
    if (particles.masses[idx] <= maxMass && mergeData.criterionValues.data[idx] > 0.0f)
    {
        const auto pos = atomicAdd(mergeData.eligibleCount, 1);
        mergeData.eligibleParticles.data[pos] = idx;
    }
}

__global__ void createMergePairs(EnhancedMergeData mergeData)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *mergeData.eligibleCount)
    {
        return;
    }

    const auto particleId = mergeData.eligibleParticles.data[idx];

    if (mergeData.states.data[particleId].status == MergeState::Status::Accepted)
    {
        const auto partnerId = mergeData.states.data[particleId].partner;
        const auto pairIdx = atomicAdd(mergeData.pairCount, 1);

        mergeData.pairs.data[pairIdx] = {particleId, partnerId, mergeData.states.data[particleId].distance, true};
    }
}

__global__ void executeMerges(ParticlesData particles,
                              EnhancedMergeData mergeData,
                              Simulation::Parameters simulationData)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *mergeData.pairCount)
    {
        return;
    }

    const auto& pair = mergeData.pairs.data[idx];
    if (!pair.valid)
    {
        return;
    }

    const auto keepIdx = pair.first;
    const auto removeIdx = pair.second;

    const auto pos1 = particles.positions[keepIdx];
    const auto pos2 = particles.positions[removeIdx];
    const auto vel1 = particles.velocities[keepIdx];
    const auto vel2 = particles.velocities[removeIdx];
    const auto mass1 = particles.masses[keepIdx];
    const auto mass2 = particles.masses[removeIdx];

    const float newMass = mass1 + mass2;
    const auto newPosition = (mass1 * pos1 + mass2 * pos2) / newMass;
    const auto newVelocity = (mass1 * vel1 + mass2 * vel2) / newMass;

    const float volumeRatio = newMass / simulationData.baseParticleMass;
    const float newRadius = simulationData.baseParticleRadius * powf(volumeRatio, 1.0f / 3.0f);
    const float newSmoothingRadius = simulationData.baseSmoothingRadius * powf(volumeRatio, 1.0f / 3.0f);

    particles.positions[keepIdx] = newPosition;
    particles.predictedPositions[keepIdx] = newPosition;
    particles.velocities[keepIdx] = newVelocity;
    particles.masses[keepIdx] = newMass;
    particles.radiuses[keepIdx] = newRadius;
    particles.smoothingRadiuses[keepIdx] = newSmoothingRadius;
    particles.densities[keepIdx] = 0.0f;
    particles.nearDensities[keepIdx] = 0.0f;
    particles.pressures[keepIdx] = 0.0f;

    mergeData.states.data[removeIdx].status = MergeState::Status::Paired;
}

__global__ void buildCompactionMap(EnhancedMergeData mergeData, uint32_t particleCount)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particleCount)
    {
        return;
    }

    mergeData.compactionMap.data[idx] = (mergeData.states.data[idx].status == MergeState::Status::Paired) ? 1 : 0;
}

__global__ void compactParticles(ParticlesData particles, EnhancedMergeData mergeData, uint32_t oldCount)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= oldCount)
    {
        return;
    }

    if (mergeData.states.data[idx].status != MergeState::Status::Paired)
    {
        const auto newIdx = idx - mergeData.compactionMap.data[idx];
        if (newIdx != idx)
        {
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
}

}
