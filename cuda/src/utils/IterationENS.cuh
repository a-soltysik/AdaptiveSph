#pragma once
#include "NeighborSharingUtils.cuh"
#include "Utils.cuh"
#include "cuda/Simulation.cuh"
#include "glm/vec4.hpp"
#include "simulation/SphSimulation.cuh"

namespace sph::cuda
{

namespace detail
{
struct SharedMemoryParticles
{
    glm::vec4* predictedPositions;
    float* smoothingRadiuses;
    float* masses;
    int32_t* originalIndices;

    __device__ void load(int32_t sharedIdx, const ParticlesData& particles, int32_t globalIdx) const;
    __device__ auto get(int32_t sharedIdx) const -> CompactParticleData;
};

__device__ auto computeWarpRectangle(const Rectangle3D& particleRect, int32_t laneIdx) -> Rectangle3D;

__device__ void loadBatchToSharedMemory(const SharedMemoryParticles& sharedParticles,
                                        const ParticlesData& particles,
                                        const SphSimulation::Grid& grid,
                                        int32_t startIdx,
                                        int32_t batchSize,
                                        int32_t laneIdx);

__device__ auto setupSharedMemoryForWarp(void* sharedMemory,
                                         int32_t warpIdxInBlock,
                                         int32_t particlesPerBatch,
                                         size_t sharedMemorySizePerBlock) -> SharedMemoryParticles;

template <typename Func>
__device__ void processNeighborIfInRange(const CompactParticleData& neighborData,
                                         const Rectangle3D& particleRect,
                                         const Simulation::Parameters& simulationData,
                                         const SphSimulation::Grid& grid,
                                         Func&& func)
{
    const auto neighborCell = calculateCellIndex(neighborData.predictedPosition, simulationData, grid);
    if (particleRect.contains(neighborCell))
    {
        std::forward<Func>(func)(neighborData.originalIndex, neighborData.predictedPosition);
    }
}

template <typename Func>
__device__ void processBatchFromSharedMemory(const SharedMemoryParticles& sharedParticles,
                                             int32_t batchSize,
                                             const Rectangle3D& particleRect,
                                             const Simulation::Parameters& simulationData,
                                             const SphSimulation::Grid& grid,
                                             Func&& func)
{
    for (auto i = 0; i < batchSize; i++)
    {
        const auto neighborData = sharedParticles.get(i);

        processNeighborIfInRange(neighborData, particleRect, simulationData, grid, std::forward<Func>(func));
    }
}

template <typename Func>
__device__ void processCellInBatches(const SphSimulation::Grid& grid,
                                     const ParticlesData& particles,
                                     const SharedMemoryParticles& sharedParticles,
                                     int32_t cellIdx,
                                     int32_t particlesPerBatch,
                                     const Rectangle3D& particleRect,
                                     const Simulation::Parameters& simulationData,
                                     int32_t laneIdx,
                                     Func&& func)
{
    const auto startIdx = grid.cellStartIndices[cellIdx];
    const auto endIdx = grid.cellEndIndices[cellIdx];

    if (startIdx == -1 || startIdx > endIdx)
    {
        return;
    }

    const auto particlesInCell = endIdx - startIdx + 1;

    for (auto batchOffset = 0; batchOffset < particlesInCell; batchOffset += particlesPerBatch)
    {
        const auto batchSize = min(particlesPerBatch, particlesInCell - batchOffset);

        loadBatchToSharedMemory(sharedParticles, particles, grid, startIdx + batchOffset, batchSize, laneIdx);
        processBatchFromSharedMemory(sharedParticles,
                                     batchSize,
                                     particleRect,
                                     simulationData,
                                     grid,
                                     std::forward<Func>(func));

        __syncwarp();
    }
}

template <typename Func>
__device__ void forEachNeighbourENS(const Rectangle3D& particleRect,
                                    const ParticlesData& particles,
                                    const SphSimulation::Grid& grid,
                                    const Simulation::Parameters& simulationData,
                                    const SharedMemoryParticles& sharedParticles,
                                    int32_t particlesPerBatch,
                                    Func&& func)
{
    const auto laneIdx = threadIdx.x % warpSize;

    const auto rectWarp = computeWarpRectangle(particleRect, laneIdx);

    for (auto z = rectWarp.min.z; z <= rectWarp.max.z; z++)
    {
        for (auto y = rectWarp.min.y; y <= rectWarp.max.y; y++)
        {
            for (auto x = rectWarp.min.x; x <= rectWarp.max.x; x++)
            {
                if (x < 0 || x >= static_cast<int32_t>(grid.gridSize.x) || y < 0 ||
                    y >= static_cast<int32_t>(grid.gridSize.y) || z < 0 || z >= static_cast<int32_t>(grid.gridSize.z))
                {
                    continue;
                }

                const auto cellIdx = flattenCellIndex({x, y, z}, grid.gridSize);
                processCellInBatches(grid,
                                     particles,
                                     sharedParticles,
                                     cellIdx,
                                     particlesPerBatch,
                                     particleRect,
                                     simulationData,
                                     laneIdx,
                                     std::forward<Func>(func));
            }
        }
    }
}

}

template <typename Func>
__device__ void forEachNeighbourENS(const Rectangle3D& particleRect,
                                    const ParticlesData& particles,
                                    const SphSimulation::Grid& grid,
                                    const Simulation::Parameters& simulationData,
                                    void* sharedMemBase,
                                    int32_t particlesPerBatch,
                                    Func&& func)
{
    const auto warpIdxInBlock = threadIdx.x / warpSize;
    const auto sharedParticles = detail::setupSharedMemoryForWarp(sharedMemBase,
                                                                  warpIdxInBlock,
                                                                  particlesPerBatch,
                                                                  simulationData.sharedMemorySizePerBlock);

    forEachNeighbourENS(particleRect,
                        particles,
                        grid,
                        simulationData,
                        sharedParticles,
                        particlesPerBatch,
                        std::forward<Func>(func));
}

}