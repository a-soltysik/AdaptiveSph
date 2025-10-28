#include <cstdint>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/ext/vector_int3.hpp>
#include <glm/ext/vector_uint3.hpp>

#include "Iteration.cuh"
#include "cuda/Simulation.cuh"

namespace sph::cuda
{

__device__ bool isOutsideGrid(const glm::ivec3& cell, const glm::uvec3& gridSize)
{
    return cell.x < 0 || cell.x >= static_cast<int32_t>(gridSize.x) || cell.y < 0 ||
           cell.y >= static_cast<int32_t>(gridSize.y) || cell.z < 0 || cell.z >= static_cast<int32_t>(gridSize.z);
}
}
