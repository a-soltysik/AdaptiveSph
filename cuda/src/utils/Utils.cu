#include <vector_types.h>

#include <cstdint>
#include <glm/common.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <utility>

#include "../simulation/SphSimulation.cuh"
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

//__device__ auto calculateCellIndex(glm::vec4 position,
//                                   const Simulation::Parameters& simulationData,
//                                   const SphSimulation::Grid& grid) -> glm::uvec3
//{
//    const auto relativePosition = glm::vec3 {position} - simulationData.domain.min;
//    const auto clampedPosition =
//        glm::clamp(relativePosition, glm::vec3(0.F), simulationData.domain.max - simulationData.domain.min);
//
//    return glm::uvec3 {clampedPosition.x / grid.cellSize.x,
//                       clampedPosition.y / grid.cellSize.y,
//                       clampedPosition.z / grid.cellSize.z};
//}

__device__ auto calculateCellIndex(glm::vec4 position,
                                   const Simulation::Parameters& simulationData,
                                   const SphSimulation::Grid& grid) -> glm::uvec3
{
    const auto relativePosition = glm::vec3 {position} - simulationData.domain.min;
    const auto domainSize = simulationData.domain.max - simulationData.domain.min;
    auto wrappedPosition = relativePosition;

    // Dla Poiseuille flow - tylko X jest periodic
    if (simulationData.testCase == Simulation::Parameters::TestCase::PoiseuilleFlow)
    {
        // POPRAWKA: Proper wrapping dla ujemnych współrzędnych
        wrappedPosition.x = wrappedPosition.x - domainSize.x * floorf(wrappedPosition.x / domainSize.x);

        // Clamp Y i Z
        wrappedPosition.y = glm::clamp(wrappedPosition.y, 0.0f, domainSize.y);
        wrappedPosition.z = glm::clamp(wrappedPosition.z, 0.0f, domainSize.z);
    }
    else if (simulationData.testCase == Simulation::Parameters::TestCase::TaylorGreenVortex)
    {
        // POPRAWKA: Wszystkie kierunki periodic
        wrappedPosition.x = wrappedPosition.x - domainSize.x * floorf(wrappedPosition.x / domainSize.x);
        wrappedPosition.y = wrappedPosition.y - domainSize.y * floorf(wrappedPosition.y / domainSize.y);
        wrappedPosition.z = wrappedPosition.z - domainSize.z * floorf(wrappedPosition.z / domainSize.z);
    }
    else
    {
        // Non-periodic cases - clamp to domain
        wrappedPosition = glm::clamp(wrappedPosition, glm::vec3(0.0f), domainSize);
    }

    auto cellIndex = glm::uvec3 {static_cast<uint32_t>(wrappedPosition.x / grid.cellSize.x),
                                 static_cast<uint32_t>(wrappedPosition.y / grid.cellSize.y),
                                 static_cast<uint32_t>(wrappedPosition.z / grid.cellSize.z)};

    // Zabezpieczenie przed out-of-bounds
    cellIndex.x = glm::min(cellIndex.x, grid.gridSize.x - 1u);
    cellIndex.y = glm::min(cellIndex.y, grid.gridSize.y - 1u);
    cellIndex.z = glm::min(cellIndex.z, grid.gridSize.z - 1u);

    return cellIndex;
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

    return {grid.cellStartIndices[neighbourCellId], grid.cellEndIndices[neighbourCellId]};
}

__device__ auto calculateMinImageDistance(const glm::vec3& pos1,
                                          const glm::vec3& pos2,
                                          const glm::vec3& domainSize,
                                          Simulation::Parameters::TestCase testCase) -> glm::vec3
{
    glm::vec3 dr = pos2 - pos1;
    // Aplikuj minimum image convention tylko dla periodic directions
    if (testCase == Simulation::Parameters::TestCase::TaylorGreenVortex)
    {
        // Wszystkie kierunki periodyczne
        dr.x = dr.x - domainSize.x * floorf(dr.x / domainSize.x + 0.5f);
        dr.y = dr.y - domainSize.y * floorf(dr.y / domainSize.y + 0.5f);
        dr.z = dr.z - domainSize.z * floorf(dr.z / domainSize.z + 0.5f);
    }
    else if (testCase == Simulation::Parameters::TestCase::PoiseuilleFlow)
    {
        // Tylko X periodic
        dr.x = dr.x - domainSize.x * floorf(dr.x / domainSize.x + 0.5f);
        // Y i Z pozostają bez zmian (solid walls)
    }
    return dr;
}

__device__ auto calculateMinImageDistance4(const glm::vec4& pos1,
                                           const glm::vec4& pos2,
                                           const glm::vec3& domainSize,
                                           Simulation::Parameters::TestCase testCase) -> glm::vec4
{
    glm::vec3 dr3 = calculateMinImageDistance(glm::vec3(pos1), glm::vec3(pos2), domainSize, testCase);
    return glm::vec4(dr3, 0.0f);
}

}
