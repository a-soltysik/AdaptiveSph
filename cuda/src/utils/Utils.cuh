#pragma once

#include <driver_types.h>
#include <vector_types.h>

#include <cstddef>
#include <cstdint>
#include <glm/ext/vector_float4.hpp>
#include <glm/ext/vector_uint3.hpp>
#include <utility>
#include <vector>

#include "../simulation/SphSimulation.cuh"
#include "cuda/Simulation.cuh"

namespace sph::cuda
{

namespace constants
{
inline constexpr auto warpSize = uint32_t {32};
}

extern __constant__ int3 offsets[27];

__device__ auto calculateCellIndex(glm::vec4 position,
                                   const Simulation::Parameters& simulationData,
                                   const SphSimulation::Grid& grid) -> glm::ivec3;
__device__ auto flattenCellIndex(glm::ivec3 cellIndex, glm::ivec3 gridSize) -> uint32_t;
__device__ auto getStartEndIndices(glm::ivec3 cellIndex, const SphSimulation::Grid& grid)
    -> std::pair<int32_t, int32_t>;

}
