#pragma once

#include <glm/ext/vector_float4.hpp>
#include <glm/ext/vector_int3.hpp>
#include <utility>

#include "cuda/Simulation.cuh"
#include "simulation/SphSimulation.cuh"

namespace sph::cuda
{

inline __device__ auto calculateCellIndex(glm::vec4 position,
                                          const Simulation::Parameters& simulationData,
                                          const SphSimulation::Grid& grid) -> glm::ivec3
{
    const auto relativePosition = glm::vec3 {position} - simulationData.domain.min;
    const auto clampedPosition =
        glm::clamp(relativePosition, glm::vec3(0.F), simulationData.domain.max - simulationData.domain.min);

    return glm::min(glm::ivec3 {clampedPosition / grid.cellSize}, grid.gridSize - 1);
}

inline __device__ auto flattenCellIndex(glm::ivec3 cellIndex, glm::ivec3 gridSize) -> uint32_t
{
    return cellIndex.x + (cellIndex.y * gridSize.x) + (cellIndex.z * gridSize.x * gridSize.y);
}

inline __device__ auto getStartEndIndices(glm::ivec3 cellIndex, const SphSimulation::Grid& grid)
    -> std::pair<int32_t, int32_t>
{
    if (cellIndex.x >= grid.gridSize.x || cellIndex.y >= grid.gridSize.y || cellIndex.z >= grid.gridSize.z)
    {
        return {-1, -1};
    }

    const auto neighbourCellId = flattenCellIndex(cellIndex, grid.gridSize);

    return {grid.cellStartIndices[neighbourCellId], grid.cellEndIndices[neighbourCellId]};
}

}
