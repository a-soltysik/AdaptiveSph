#include <device_atomic_functions.h>

#include <array>
#include <cstdint>
#include <glm/exponential.hpp>
#include <glm/geometric.hpp>

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

__global__ void mergeParticles(ParticlesData particles, RefinementData refinementData)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *refinementData.merge.particlesMergeCount)
    {
        return;
    }
    const auto keepIdx = refinementData.merge.particlesIdsToMerge.first.data[idx];
    const auto removeIdx = refinementData.merge.particlesIdsToMerge.second.data[idx];

    if (removeIdx == UINT_MAX || keepIdx == removeIdx || keepIdx >= particles.particleCount ||
        removeIdx >= particles.particleCount)
    {
        return;
    }

    if (refinementData.merge.removalFlags.data[keepIdx] == RefinementData::RemovalState::Default)
    {
        return;
    }

    const auto positions = std::pair {particles.positions[keepIdx], particles.positions[removeIdx]};
    const auto velocities = std::pair {particles.velocities[keepIdx], particles.velocities[removeIdx]};
    const auto masses = std::pair {particles.masses[keepIdx], particles.masses[removeIdx]};
    const auto radiuses = std::pair {particles.radiuses[keepIdx], particles.radiuses[removeIdx]};
    const auto smoothingRadiuses =
        std::pair {particles.smoothingRadiuses[keepIdx], particles.smoothingRadiuses[removeIdx]};

    const float newMass = masses.first + masses.second;
    const float newRadius =
        getNewRadius(newMass, (masses.first + masses.second) / 2.0F, (radiuses.first + radiuses.second) / 2.0F);

    const auto newPosition = (masses.first * positions.first + masses.second * positions.second) / newMass;
    const auto newVelocity = (masses.first * velocities.first + masses.second * velocities.second) / newMass;

    const auto distances =
        std::pair {glm::length(newPosition - positions.first), glm::length(newPosition - positions.second)};

    const auto kernelValues = std::pair {device::densityKernel(distances.first, smoothingRadiuses.first),
                                         device::densityKernel(distances.second, smoothingRadiuses.second)};

    const auto denominator = (masses.first * kernelValues.first) + (masses.second * kernelValues.second);

    const auto newSmoothingLength = glm::pow(15.F / (glm::pi<float>() * 2.F) * newMass / denominator, 1.0F / 3.0F);

    particles.positions[keepIdx] = newPosition;
    particles.predictedPositions[keepIdx] = newPosition;
    particles.velocities[keepIdx] = newVelocity;
    particles.masses[keepIdx] = newMass;
    particles.radiuses[keepIdx] = newRadius;
    particles.smoothingRadiuses[keepIdx] = newSmoothingLength;
    particles.densities[keepIdx] = 0.0F;
    particles.nearDensities[keepIdx] = 0.0F;
    particles.pressures[keepIdx] = 0.0F;
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
        uint32_t removedCount = 0;
        for (uint32_t i = 0; i < particleCount; i++)
        {
            if (refinementData.merge.removalFlags.data[i] == RefinementData::RemovalState::Remove)
            {
                removedCount++;
            }
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

    if (second == UINT_MAX || first == second || first >= particles.particleCount || second >= particles.particleCount)
    {
        return;
    }

    const auto keepIdx = min(first, second);
    const auto removeIdx = max(first, second);

    const auto firstResult = atomicCAS(reinterpret_cast<int*>(&refinementData.merge.removalFlags.data[keepIdx]), 0, 1);
    if (firstResult != 0)
    {
        return;
    }

    const auto secondResult =
        atomicCAS(reinterpret_cast<int*>(&refinementData.merge.removalFlags.data[removeIdx]), 0, 2);

    if (secondResult != 0)
    {
        atomicExch(reinterpret_cast<int*>(&refinementData.merge.removalFlags.data[keepIdx]), 0);
        return;
    }

    refinementData.merge.particlesIdsToMerge.first.data[idx] = keepIdx;
    refinementData.merge.particlesIdsToMerge.second.data[idx] = removeIdx;
}

}
