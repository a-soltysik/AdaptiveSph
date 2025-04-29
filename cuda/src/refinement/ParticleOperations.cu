#include <device_atomic_functions.h>

#include <array>
#include <glm/geometric.hpp>

#include "../common/Utils.cuh"
#include "../device/Kernel.cuh"
#include "ParticleOperations.cuh"
#include "glm/ext/scalar_constants.hpp"

namespace sph::cuda::refinement
{
namespace detail
{
__device__ __constant__ float phi = 1.61803398875f;
__device__ __constant__ float invnorm = 0.5257311121f;
}  // namespace detail

__device__ float getNewRadius(float mass, float baseMass, float baseRadius)
{
    return baseRadius * glm::pow(mass / baseMass, 1.0f / 3.0f);
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
        particles.positions[newIdx] = originalPosition + glm::vec4(offset, 0.0f);
        particles.predictedPositions[newIdx] = particles.positions[newIdx];
        particles.velocities[newIdx] = originalVelocity;
        particles.masses[newIdx] = daughterMass;
        particles.radiuses[newIdx] = newRadius;
        particles.smoothingRadiuses[newIdx] = newSmoothingLength;

        particles.densities[newIdx] = 0.0f;
        particles.nearDensities[newIdx] = 0.0f;
        particles.pressures[newIdx] = 0.0f;
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
    // Skip invalid pairs
    if (removeIdx == UINT_MAX || keepIdx == removeIdx || keepIdx >= particles.particleCount ||
        removeIdx >= particles.particleCount)
    {
        return;
    }
    // Double-check that particles are properly marked
    // This prevents race conditions if markPotentialMerges had conflicts
    if (refinementData.merge.removalFlags.data[keepIdx] != 1 || refinementData.merge.removalFlags.data[removeIdx] != 2)
    {
        return;
    }

    // Get properties of both particles
    const auto positions = std::pair {particles.positions[keepIdx], particles.positions[removeIdx]};
    const auto velocities = std::pair {particles.velocities[keepIdx], particles.velocities[removeIdx]};
    const auto masses = std::pair {particles.masses[keepIdx], particles.masses[removeIdx]};
    const auto radiuses = std::pair {particles.radiuses[keepIdx], particles.radiuses[removeIdx]};
    const auto smoothingRadiuses =
        std::pair {particles.smoothingRadiuses[keepIdx], particles.smoothingRadiuses[removeIdx]};

    // Validate data
    if (isnan(positions.first.x) || isnan(positions.second.x) || isnan(velocities.first.x) ||
        isnan(velocities.second.x) || isnan(masses.first) || isnan(masses.second) || masses.first <= 0.0f ||
        masses.second <= 0.0f)
    {
        // Invalid data - unmark and skip
        atomicExch(reinterpret_cast<int*>(&refinementData.merge.removalFlags.data[keepIdx]), 0);
        atomicExch(reinterpret_cast<int*>(&refinementData.merge.removalFlags.data[removeIdx]), 0);
        return;
    }

    // Calculate new properties
    const float newMass = masses.first + masses.second;
    const float newRadius =
        getNewRadius(newMass, (masses.first + masses.second) / 2.0f, (radiuses.first + radiuses.second) / 2.0f);

    // Mass-weighted position and velocity (conserve momentum)
    const auto newPosition = (masses.first * positions.first + masses.second * positions.second) / newMass;
    const auto newVelocity = (masses.first * velocities.first + masses.second * velocities.second) / newMass;

    // Calculate smoothing length based on Vacondio et al. paper
    const auto distances =
        std::pair {glm::length(newPosition - positions.first), glm::length(newPosition - positions.second)};

    const auto kernelValues = std::pair {device::densityKernel(distances.first, smoothingRadiuses.first),
                                         device::densityKernel(distances.second, smoothingRadiuses.second)};

    float denominator = masses.first * kernelValues.first + masses.second * kernelValues.second;

    const auto newSmoothingLength = glm::pow(15.F / (glm::pi<float>() * 2.F) * newMass / denominator, 1.0f / 3.0f);

    // Update the kept particle with new properties
    particles.positions[keepIdx] = newPosition;
    particles.predictedPositions[keepIdx] = newPosition;
    particles.velocities[keepIdx] = newVelocity;
    particles.masses[keepIdx] = newMass;
    particles.radiuses[keepIdx] = newRadius;
    particles.smoothingRadiuses[keepIdx] = newSmoothingLength;
    particles.densities[keepIdx] = 0.0f;
    particles.nearDensities[keepIdx] = 0.0f;
    particles.pressures[keepIdx] = 0.0f;
}

__global__ void computeDensitiesWithVariableSmoothingLengths(ParticlesData particles,
                                                             SphSimulation::State state,
                                                             Simulation::Parameters simulationData)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }

    const auto position = particles.predictedPositions[idx];
    const auto originCell = calculateCellIndex(position, simulationData, state.grid);
    auto density = 0.F;
    auto nearDensity = 0.F;

    for (const auto offset : offsets)
    {
        const auto neighborCell = originCell + glm::uvec3 {offset.x, offset.y, offset.z};
        if (neighborCell.x >= state.grid.gridSize.x || neighborCell.y >= state.grid.gridSize.y ||
            neighborCell.z >= state.grid.gridSize.z)
        {
            continue;
        }

        const auto neighbourCellId = flattenCellIndex(neighborCell, state.grid.gridSize);
        const auto startIdx = state.grid.cellStartIndices.data[neighbourCellId];
        const auto endIdx = state.grid.cellEndIndices.data[neighbourCellId];

        if (startIdx == -1 || startIdx > endIdx)
        {
            continue;
        }

        for (auto i = startIdx; i <= endIdx; i++)
        {
            const auto neighbourIdx = state.grid.particleArrayIndices.data[i];
            const auto neighbourPosition = particles.predictedPositions[neighbourIdx];
            const auto offsetToNeighbour = neighbourPosition - position;
            const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);

            // Get neighbor's smoothing radius
            const auto neighbourSmoothingLength = particles.smoothingRadiuses[neighbourIdx];

            // Use scatter formulation: evaluate kernel using neighbor's smoothing radius
            const auto maxRadiusSquared = neighbourSmoothingLength * neighbourSmoothingLength;

            if (distanceSquared > maxRadiusSquared)
            {
                continue;
            }

            const auto distance = glm::sqrt(distanceSquared);
            const auto neighbourMass = particles.masses[neighbourIdx];

            // Use neighbor's smoothing radius in kernel evaluation (scatter approach)
            //density += neighbourMass * device::densityKernel(distance, neighbourSmoothingLength);
            //nearDensity += neighbourMass * device::nearDensityKernel(distance, neighbourSmoothingLength);

            density += neighbourMass * device::densityKernel(distance, particles.smoothingRadiuses[idx]);
            nearDensity += neighbourMass * device::nearDensityKernel(distance, particles.smoothingRadiuses[idx]);
        }
    }

    particles.densities[idx] = density;
    particles.nearDensities[idx] = nearDensity;
}

__global__ void computePressureForceWithVariableSmoothingLengths(ParticlesData particles,
                                                                 SphSimulation::State state,
                                                                 Simulation::Parameters simulationData,
                                                                 float dt)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }
    const auto position = particles.predictedPositions[idx];
    const auto density = particles.densities[idx];
    const auto nearDensity = particles.nearDensities[idx];
    const auto pressure = (density - simulationData.restDensity) * simulationData.pressureConstant;
    const auto nearPressure = nearDensity * simulationData.nearPressureConstant;
    // Get particle's own smoothing radius
    const auto particleRadius = particles.radiuses[idx];

    auto pressureForce = glm::vec4 {};

    const auto originCell = calculateCellIndex(position, simulationData, state.grid);

    for (const auto offset : offsets)
    {
        const auto neighborCell = originCell + glm::uvec3 {offset.x, offset.y, offset.z};
        if (neighborCell.x >= state.grid.gridSize.x || neighborCell.y >= state.grid.gridSize.y ||
            neighborCell.z >= state.grid.gridSize.z)
        {
            continue;
        }
        const auto neighbourCellId = flattenCellIndex(neighborCell, state.grid.gridSize);
        const auto startIdx = state.grid.cellStartIndices.data[neighbourCellId];
        const auto endIdx = state.grid.cellEndIndices.data[neighbourCellId];

        if (startIdx == -1)
        {
            continue;
        }

        for (auto i = startIdx; i <= endIdx; i++)
        {
            const auto neighbourIdx = state.grid.particleArrayIndices.data[i];
            if (neighbourIdx == idx)
            {
                continue;
            }

            const auto neighbourPosition = particles.predictedPositions[neighbourIdx];
            const auto offsetToNeighbour = neighbourPosition - position;
            const auto distanceSquared = glm::dot(offsetToNeighbour, offsetToNeighbour);
            // Get neighbor's smoothing radius
            const auto neighbourRadius = particles.radiuses[neighbourIdx];
            // Check both particles' interaction radii
            const auto maxRadiusSquared = fmaxf(particleRadius * particleRadius, neighbourRadius * neighbourRadius);
            if (distanceSquared > maxRadiusSquared)
            {
                continue;
            }
            const auto densityNeighbour = particles.densities[neighbourIdx];
            const auto nearDensityNeighbour = particles.nearDensities[neighbourIdx];
            const auto pressureNeighbour =
                (densityNeighbour - simulationData.restDensity) * simulationData.pressureConstant;
            const auto nearPressureNeighbour = nearDensityNeighbour * simulationData.nearPressureConstant;

            const auto sharedPressure = (pressure + pressureNeighbour) / 2.F;
            const auto sharedNearPressure = (nearPressure + nearPressureNeighbour) / 2.F;

            const auto distance = glm::sqrt(distanceSquared);
            const auto direction = distance > 0.F ? offsetToNeighbour / distance : glm::vec4(0.F, 1.F, 0.F, 0.F);

            const auto neighbourMass = particles.masses[neighbourIdx];

            // Implement variationally consistent SPH formulation for variable smoothing lengths
            // This correctly handles momentum conservation with variable particle sizes
            // Calculate pressure gradient using neighbor's smoothing radius
            //pressureForce += neighbourMass * direction * device::densityDerivativeKernel(distance, neighbourRadius) *
            //                 sharedPressure / densityNeighbour;
            //// Calculate pressure gradient using particle's smoothing radius
            //pressureForce -= neighbourMass * direction * device::densityDerivativeKernel(distance, particleRadius) *
            //                 sharedPressure / densityNeighbour;

            //// Similar terms for near pressure
            //pressureForce += neighbourMass * direction *
            //                 device::nearDensityDerivativeKernel(distance, neighbourRadius) * sharedNearPressure /
            //                 nearDensityNeighbour;

            //pressureForce -= neighbourMass * direction * device::nearDensityDerivativeKernel(distance, particleRadius) *
            //                 sharedNearPressure / nearDensityNeighbour;

            pressureForce += neighbourMass * direction * device::densityDerivativeKernel(distance, particleRadius) *
                             sharedPressure / densityNeighbour;
            pressureForce += neighbourMass * direction * device::nearDensityDerivativeKernel(distance, particleRadius) *
                             sharedNearPressure / nearDensityNeighbour;
        }
    }

    const auto particleMass = particles.masses[idx];
    const auto acceleration = pressureForce / (density * particleMass);

    particles.velocities[idx] += acceleration * dt;
}

__device__ std::pair<uint32_t, float> findClosestParticle(const ParticlesData& particles,
                                                          uint32_t particleIdx,
                                                          const SphSimulation::Grid& grid,
                                                          const Simulation::Parameters& simulationData)
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

    return {result.first, sqrtf(result.second)};
}

__global__ void removeParticles(ParticlesData particles, RefinementData refinementData)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }

    // Only process particles that aren't marked for removal (with value 2)
    // Particles marked with 1 are kept and were part of a merge operation
    if (refinementData.merge.removalFlags.data[idx] != 2)
    {
        const auto newId = idx - refinementData.merge.prefixSums.data[idx];
        if (newId != idx)
        {
            // Copy particle data to new position
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
    // Find the closest particle using existing helper function
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
            if (refinementData.merge.removalFlags.data[i] == 2)
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
    // Skip invalid pairs
    if (second == UINT_MAX || first == second || first >= particles.particleCount || second >= particles.particleCount)
    {
        return;
    }
    // Ensure we process pairs in a consistent order (smaller ID first)
    // This helps prevent deadlocks
    const auto keepIdx = min(first, second);
    const auto removeIdx = max(first, second);
    // Try to mark the first particle as "keep" (value 1)
    const auto firstResult = atomicCAS(reinterpret_cast<int*>(&refinementData.merge.removalFlags.data[keepIdx]), 0, 1);
    // If first particle already marked, abort this merge
    if (firstResult != 0)
    {
        return;
    }
    // Try to mark the second particle as "remove" (value 2)
    const auto secondResult =
        atomicCAS(reinterpret_cast<int*>(&refinementData.merge.removalFlags.data[removeIdx]), 0, 2);

    // If second particle already marked, unmark the first and abort
    if (secondResult != 0)
    {
        atomicExch(reinterpret_cast<int*>(&refinementData.merge.removalFlags.data[keepIdx]), 0);
        return;
    }

    // Successfully marked both particles
    // Update the merge pair info for the performMarkedMerges kernel
    refinementData.merge.particlesIdsToMerge.first.data[idx] = keepIdx;
    refinementData.merge.particlesIdsToMerge.second.data[idx] = removeIdx;
}

}
