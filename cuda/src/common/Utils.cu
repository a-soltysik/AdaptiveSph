#include "../SphSimulation.cuh"
#include "Utils.cuh"
#include "cuda/Simulation.cuh"
#include "glm/ext/vector_uint3.hpp"

namespace sph::cuda
{

__constant__ int3 offsets[27] = {
    {0,  0,  0 },
    {0,  0,  -1},
    {0,  0,  1 },
    {0,  -1, 0 },
    {0,  -1, -1},
    {0,  -1, 1 },
    {0,  1,  0 },
    {0,  1,  -1},
    {0,  1,  1 },
    {-1, 0,  0 },
    {-1, 0,  -1},
    {-1, 0,  1 },
    {-1, -1, 0 },
    {-1, -1, -1},
    {-1, -1, 1 },
    {-1, 1,  0 },
    {-1, 1,  -1},
    {-1, 1,  1 },
    {1,  0,  0 },
    {1,  0,  -1},
    {1,  0,  1 },
    {1,  -1, 0 },
    {1,  -1, -1},
    {1,  -1, 1 },
    {1,  1,  0 },
    {1,  1,  -1},
    {1,  1,  1 }
};

__device__ auto calculateCellIndex(glm::vec4 position,
                                   const Simulation::Parameters& simulationData,
                                   const SphSimulation::Grid& grid) -> glm::uvec3
{
    const auto relativePosition = glm::vec3 {position} - simulationData.domain.min;
    const auto clampedPosition =
        glm::clamp(relativePosition, glm::vec3(0.F), simulationData.domain.max - simulationData.domain.min);

    return glm::uvec3 {clampedPosition.x / grid.cellSize.x,
                       clampedPosition.y / grid.cellSize.y,
                       clampedPosition.z / grid.cellSize.z};
}

__device__ auto flattenCellIndex(glm::uvec3 cellIndex, glm::uvec3 gridSize) -> uint32_t
{
    return cellIndex.x + (cellIndex.y * gridSize.x) + (cellIndex.z * gridSize.x * gridSize.y);
}

__device__ auto getStartEndIndices(glm::uvec3 cellIndex, const SphSimulation::Grid& grid) -> std::pair<int32_t, int32_t>
{
    if (cellIndex.x >= grid.gridSize.x || cellIndex.y >= grid.gridSize.y || cellIndex.z >= grid.gridSize.z)
    {
        return {-1, -1};
    }

    const auto neighbourCellId = flattenCellIndex(cellIndex, grid.gridSize);

    return {grid.cellStartIndices.data[neighbourCellId], grid.cellEndIndices.data[neighbourCellId]};
}

__device__ float computePeriodicDistanceSquared(const glm::vec4& pos1,
                                                const glm::vec4& pos2,
                                                const cuda::Simulation::Parameters& simulationData,
                                                glm::vec4* adjustedOffset)
{
    glm::vec4 offset = pos2 - pos1;
    // Check if we should use periodic boundary in X direction
    if (simulationData.testCase == cuda::Simulation::Parameters::TestCase::PoiseuilleFlow)
    {
        const float domainWidth = simulationData.domain.max.x - simulationData.domain.min.x;

        // If distance in X axis is greater than half the domain width,
        // it's shorter to wrap around
        if (fabsf(offset.x) > domainWidth * 0.5f)
        {
            if (offset.x > 0)
            {
                offset.x -= domainWidth;
            }
            else
            {
                offset.x += domainWidth;
            }
        }
    }

    // If caller wants the adjusted offset, provide it
    if (adjustedOffset != nullptr)
    {
        *adjustedOffset = offset;
    }

    // Return the squared Euclidean distance
    return glm::dot(glm::vec3(offset), glm::vec3(offset));
}
}
