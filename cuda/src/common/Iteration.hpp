#pragma once
#include "../SphSimulation.cuh"
#include "Utils.cuh"

namespace sph::cuda
{

__device__ void handlePeriodicBoundaries(const glm::ivec3& targetCell,
                                         glm::ivec3& wrappedCell,
                                         glm::vec3& positionShift,
                                         const glm::uvec3& gridSize,
                                         const glm::vec3& domainSize,
                                         Simulation::Parameters::TestCase testCase);

__device__ void applyPeriodicBoundary(
    int targetCoord, int& wrappedCoord, float& positionShift, uint32_t gridSize, float domainSize);

__device__ bool isOutsideGrid(const glm::ivec3& cell, const glm::uvec3& gridSize);

__device__ void applyMinimumImageConvention(const glm::vec4& position,
                                            glm::vec4& shiftedPos,
                                            const glm::vec3& domainSize,
                                            Simulation::Parameters::TestCase testCase);

__device__ void applyMinimumImageShift(float diff, float& coord, float domainSize);

template <typename Func>
__device__ void forEachNeighbour(glm::vec4 position,
                                 const ParticlesData& particles,
                                 const SphSimulation::Parameters& simulationData,
                                 const SphSimulation::Grid& grid,
                                 Func&& func)
{
    // Whether to use periodic boundaries
    bool usePeriodic = (simulationData.testCase == Simulation::Parameters::TestCase::PoiseuilleFlow ||
                        simulationData.testCase == Simulation::Parameters::TestCase::TaylorGreenVortex);

    // Domain dimensions
    const glm::vec3 domainMin = simulationData.domain.min;
    const glm::vec3 domainMax = simulationData.domain.max;
    const glm::vec3 domainSize = domainMax - domainMin;

    // Get the base cell for the current particle
    const auto centerCell = calculateCellIndex(position, simulationData, grid);

    // Process each neighboring cell (including the particle's own cell)
    for (const auto& offset : offsets)
    {
        // Apply offset to get neighboring cell
        glm::ivec3 targetCell(static_cast<int>(centerCell.x) + offset.x,
                              static_cast<int>(centerCell.y) + offset.y,
                              static_cast<int>(centerCell.z) + offset.z);

        // Wrap cells for periodic boundaries
        glm::ivec3 wrappedCell = targetCell;
        glm::vec3 positionShift(0.0f);

        if (usePeriodic)
        {
            // Handle wrapping based on simulation type
            handlePeriodicBoundaries(targetCell,
                                     wrappedCell,
                                     positionShift,
                                     grid.gridSize,
                                     domainSize,
                                     simulationData.testCase);
        }
        else
        {
            // For non-periodic boundaries, skip cells outside the grid
            if (isOutsideGrid(targetCell, grid.gridSize))
            {
                continue;
            }
        }

        // Skip if still outside grid after wrapping
        if (isOutsideGrid(wrappedCell, grid.gridSize))
        {
            continue;
        }

        // Convert to unsigned for flattening
        glm::uvec3 wrappedCellU(wrappedCell.x, wrappedCell.y, wrappedCell.z);

        // Get particle range in this cell
        const auto cellIdx = flattenCellIndex(wrappedCellU, grid.gridSize);
        const auto startIdx = grid.cellStartIndices.data[cellIdx];
        const auto endIdx = grid.cellEndIndices.data[cellIdx];

        if (startIdx == -1 || startIdx > endIdx)
        {
            continue;  // Skip empty cells
        }

        // Process each particle in this cell
        for (auto i = startIdx; i <= endIdx; i++)
        {
            const auto neighborIdx = grid.particleArrayIndices.data[i];
            const auto neighborPos = particles.predictedPositions[neighborIdx];
            // Apply the position shift for periodic boundaries
            glm::vec4 shiftedPos = neighborPos;

            if (glm::length(positionShift) > 1e-6f)
            {
                // If we crossed a boundary, shift the particle position
                shiftedPos = neighborPos + glm::vec4(positionShift, 0.0f);
            }
            else if (usePeriodic)
            {
                // Apply minimum image convention
                applyMinimumImageConvention(position, shiftedPos, domainSize, simulationData.testCase);
            }

            // Pass the neighbor index and its adjusted position to the callback
            func(neighborIdx, shiftedPos);
        }
    }
}
}
