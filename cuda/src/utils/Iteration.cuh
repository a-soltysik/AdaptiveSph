#pragma once

#include <cstdint>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/ext/vector_int3.hpp>
#include <glm/ext/vector_uint3.hpp>

#include "../simulation/SphSimulation.cuh"
#include "Utils.cuh"
#include "cuda/Simulation.cuh"

namespace sph::cuda
{

__device__ void handlePeriodicBoundaries(const glm::ivec3& targetCell,
                                         glm::ivec3& wrappedCell,
                                         glm::vec3& positionShift,
                                         const glm::uvec3& gridSize,
                                         const glm::vec3& domainSize,
                                         Simulation::Parameters::TestCase testCase);

__device__ void applyPeriodicBoundary(
    int32_t targetCoord, int32_t& wrappedCoord, float& positionShift, uint32_t gridSize, float domainSize);

__device__ bool isOutsideGrid(const glm::ivec3& cell, const glm::uvec3& gridSize);

__device__ void adjustPositionForWrapping(const glm::vec4& position,
                                          glm::vec4& shiftedPos,
                                          const glm::vec3& domainSize,
                                          Simulation::Parameters::TestCase testCase);

__device__ void shiftCoordinateForWrapping(float diff, float& coord, float domainSize);

template <typename Func>
__device__ void forEachNeighbour(glm::vec4 position,
                                 const ParticlesData& particles,
                                 const SphSimulation::Parameters& simulationData,
                                 const SphSimulation::Grid& grid,
                                 Func&& func)
{
    const auto usePeriodic = (simulationData.testCase == Simulation::Parameters::TestCase::PoiseuilleFlow ||
                              simulationData.testCase == Simulation::Parameters::TestCase::TaylorGreenVortex);

    const auto domainMin = simulationData.domain.min;
    const auto domainMax = simulationData.domain.max;
    const auto domainSize = domainMax - domainMin;

    const auto centerCell = calculateCellIndex(position, simulationData, grid);

    for (const auto& offset : offsets)
    {
        const auto targetCell = glm::ivec3 {static_cast<int32_t>(centerCell.x) + offset.x,
                                            static_cast<int32_t>(centerCell.y) + offset.y,
                                            static_cast<int32_t>(centerCell.z) + offset.z};

        glm::ivec3 wrappedCell = targetCell;
        glm::vec3 positionShift(0.0F);

        if (usePeriodic)
        {
            handlePeriodicBoundaries(targetCell,
                                     wrappedCell,
                                     positionShift,
                                     grid.gridSize,
                                     domainSize,
                                     simulationData.testCase);
        }

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

            // KLUCZOWA ZMIANA: Użyj minimum image distance zamiast position shifting
            glm::vec4 adjustedNeighborPos;
            if (usePeriodic)
            {
                adjustedNeighborPos =
                    position + calculateMinImageDistance4(position, neighborPos, domainSize, simulationData.testCase);
            }
            else
            {
                adjustedNeighborPos = neighborPos;
            }

            func(neighborIdx, adjustedNeighborPos);
        }
    }
}
}
