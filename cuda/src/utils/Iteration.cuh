#pragma once

#include <cstdint>
#include <glm/ext/vector_float4.hpp>
#include <glm/ext/vector_int3.hpp>
#include <glm/ext/vector_uint3.hpp>

#include "../simulation/SphSimulation.cuh"
#include "Utils.cuh"
#include "cuda/Simulation.cuh"

namespace sph::cuda
{

__device__ bool isOutsideGrid(const glm::ivec3& cell, const glm::uvec3& gridSize);

template <typename Func>
__device__ void forEachNeighbour(glm::vec4 position,
                                 const ParticlesData& particles,
                                 const SphSimulation::Parameters& simulationData,
                                 const SphSimulation::Grid& grid,
                                 Func&& func)
{
    const auto centerCell = calculateCellIndex(position, simulationData, grid);

    for (const auto& offset : offsets)
    {
        const auto targetCell = glm::ivec3 {static_cast<int32_t>(centerCell.x) + offset.x,
                                            static_cast<int32_t>(centerCell.y) + offset.y,
                                            static_cast<int32_t>(centerCell.z) + offset.z};

        glm::ivec3 wrappedCell = targetCell;

        if (isOutsideGrid(wrappedCell, grid.gridSize))
        {
            continue;
        }

        const auto cellIdx = flattenCellIndex(glm::uvec3 {wrappedCell.x, wrappedCell.y, wrappedCell.z}, grid.gridSize);
        const auto startIdx = grid.cellStartIndices[cellIdx];
        const auto endIdx = grid.cellEndIndices[cellIdx];

        if (startIdx == -1 || startIdx > endIdx)
        {
            continue;
        }

        for (auto i = startIdx; i <= endIdx; i++)
        {
            const auto neighborIdx = grid.particleArrayIndices[i];
            const auto neighborPos = particles.predictedPositions[neighborIdx];
            func(neighborIdx, neighborPos);
        }
    }
}
}
