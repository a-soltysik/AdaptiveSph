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

extern __constant__ int3 offsets[27];

__device__ auto calculateCellIndex(glm::vec4 position,
                                   const Simulation::Parameters& simulationData,
                                   const SphSimulation::Grid& grid) -> glm::uvec3;
__device__ auto flattenCellIndex(glm::uvec3 cellIndex, glm::uvec3 gridSize) -> uint32_t;
__device__ auto getStartEndIndices(glm::uvec3 cellIndex, const SphSimulation::Grid& grid)
    -> std::pair<int32_t, int32_t>;

template <typename T>
auto fromGpu(const T* gpuPtr) -> T
{
    auto hostData = T {};
    cudaMemcpy(&hostData, gpuPtr, sizeof(T), cudaMemcpyDeviceToHost);
    return hostData;
}

template <typename T>
auto fromGpu(const T* gpuPtr, size_t elementsCount) -> std::vector<T>
{
    auto hostData = std::vector<T>(elementsCount);
    cudaMemcpy(hostData.data(), gpuPtr, elementsCount * sizeof(T), cudaMemcpyDeviceToHost);
    return hostData;
}

__device__ auto calculateMinImageDistance(const glm::vec3& pos1,
                                          const glm::vec3& pos2,
                                          const glm::vec3& domainSize,
                                          Simulation::Parameters::TestCase testCase) -> glm::vec3;

__device__ auto calculateMinImageDistance4(const glm::vec4& pos1,
                                           const glm::vec4& pos2,
                                           const glm::vec3& domainSize,
                                           Simulation::Parameters::TestCase testCase) -> glm::vec4;
}
