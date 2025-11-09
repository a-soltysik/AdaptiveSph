#pragma once

#include "IterationENS.cuh"

namespace sph::cuda::detail
{

__device__ void SharedMemoryParticles::load(int32_t sharedIdx, const ParticlesData& particles, int32_t globalIdx) const
{
    predictedPositions[sharedIdx] = particles.predictedPositions[globalIdx];
    smoothingRadiuses[sharedIdx] = particles.smoothingRadiuses[globalIdx];
    masses[sharedIdx] = particles.masses[globalIdx];
    originalIndices[sharedIdx] = globalIdx;
}

__device__ CompactParticleData SharedMemoryParticles::get(int32_t sharedIdx) const
{
    return {
        .predictedPosition = predictedPositions[sharedIdx],
        .smoothingRadius = smoothingRadiuses[sharedIdx],
        .mass = masses[sharedIdx],
        .originalIndex = originalIndices[sharedIdx],
    };
}

__device__ auto computeWarpRectangle(const Rectangle3D& particleRect, int32_t laneIdx) -> Rectangle3D
{
    auto localMinX = particleRect.min.x;
    auto localMinY = particleRect.min.y;
    auto localMinZ = particleRect.min.z;
    auto localMaxX = particleRect.max.x;
    auto localMaxY = particleRect.max.y;
    auto localMaxZ = particleRect.max.z;

    const auto mask = 0xFFFFFFFF;
    for (auto offset = constants::warpSize / 2; offset > 0; offset /= 2)
    {
        localMinX = min(localMinX, __shfl_down_sync(mask, localMinX, offset));
        localMinY = min(localMinY, __shfl_down_sync(mask, localMinY, offset));
        localMinZ = min(localMinZ, __shfl_down_sync(mask, localMinZ, offset));
        localMaxX = max(localMaxX, __shfl_down_sync(mask, localMaxX, offset));
        localMaxY = max(localMaxY, __shfl_down_sync(mask, localMaxY, offset));
        localMaxZ = max(localMaxZ, __shfl_down_sync(mask, localMaxZ, offset));
    }

    auto result = Rectangle3D {};
    if (laneIdx == 0)
    {
        result = {
            .min = {localMinX, localMinY, localMinZ},
            .max = {localMaxX, localMaxY, localMaxZ}
        };
    }

    return {
        .min = {__shfl_sync(mask, result.min.x, 0),
                __shfl_sync(mask, result.min.y, 0),
                __shfl_sync(mask, result.min.z, 0)},
        .max = {__shfl_sync(mask, result.max.x, 0),
                __shfl_sync(mask, result.max.y, 0),
                __shfl_sync(mask, result.max.z, 0)}
    };
}

__device__ void loadBatchToSharedMemory(const SharedMemoryParticles& sharedParticles,
                                        const ParticlesData& particles,
                                        const SphSimulation::Grid& grid,
                                        int32_t startIdx,
                                        int32_t batchSize,
                                        int32_t laneIdx)
{
    for (auto i = laneIdx; i < batchSize; i += warpSize)
    {
        const auto globalParticleIdx = grid.particleArrayIndices[startIdx + i];
        sharedParticles.load(i, particles, globalParticleIdx);
    }

    __syncwarp();
}

__device__ SharedMemoryParticles setupSharedMemoryForWarp(void* sharedMemory,
                                                          int32_t warpIdxInBlock,
                                                          int32_t particlesPerBatch,
                                                          size_t sharedMemorySizePerBlock)
{
    const auto warpsPerBlock = (blockDim.x + constants::warpSize - 1) / constants::warpSize;
    const auto sharedMemoryPerWarp = sharedMemorySizePerBlock / warpsPerBlock;

    auto* warpBase = static_cast<char*>(sharedMemory) + warpIdxInBlock * sharedMemoryPerWarp;

    const auto positionsSize = particlesPerBatch * sizeof(glm::vec4);
    const auto radiusesSize = particlesPerBatch * sizeof(float);
    const auto massesSize = particlesPerBatch * sizeof(float);

    auto* positions = reinterpret_cast<glm::vec4*>(warpBase);
    auto* smoothingRadiuses = reinterpret_cast<float*>(warpBase + positionsSize);
    auto* masses = reinterpret_cast<float*>(warpBase + positionsSize + radiusesSize);
    auto* indices = reinterpret_cast<int32_t*>(warpBase + positionsSize + radiusesSize + massesSize);

    return {
        .predictedPositions = positions,
        .smoothingRadiuses = smoothingRadiuses,
        .masses = masses,
        .originalIndices = indices,
    };
}

}