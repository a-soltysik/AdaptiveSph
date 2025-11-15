#pragma once

#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/ext/vector_int3.hpp>

#include "Utils.cuh"
#include "cuda/Simulation.cuh"
#include "simulation/SphSimulation.cuh"

namespace sph::cuda
{

template <typename Func>
__device__ void forEachNeighbour(glm::vec4 position,
                                 const glm::vec4* positions,
                                 const SphSimulation::Parameters::Domain& domain,
                                 const SphSimulation::Grid& grid,
                                 float smoothingRadius,
                                 Func&& func)
{
    const auto min = glm::max(glm::ivec3 {(glm::vec3 {position} - domain.min - smoothingRadius) / grid.cellSize},
                              glm::ivec3 {0, 0, 0});

    const auto max =
        glm::min(glm::ivec3 {(glm::vec3 {position} - domain.min + smoothingRadius) / grid.cellSize}, grid.gridSize - 1);

    for (auto x = min.x; x <= max.x; x++)
    {
        for (auto y = min.y; y <= max.y; y++)
        {
            for (auto z = min.z; z <= max.z; z++)
            {
                const auto cellIdx = flattenCellIndex(glm::ivec3 {x, y, z}, grid.gridSize);
                const auto startIdx = grid.cellStartIndices[cellIdx];
                const auto endIdx = grid.cellEndIndices[cellIdx];

                if (startIdx == -1 || startIdx > endIdx)
                {
                    continue;
                }

                for (auto i = startIdx; i <= endIdx; i++)
                {
                    const auto neighborIdx = grid.particleArrayIndices[i];
                    const auto neighborPos = positions[neighborIdx];
                    func(neighborIdx, neighborPos);
                }
            }
        }
    }
}

}
