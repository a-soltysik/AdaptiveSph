#pragma once
#include "../SphSimulation.cuh"
#include "Utils.cuh"

namespace sph::cuda
{

template <typename Func>
__device__ void forEachNeighbour(glm::vec4 position,
                                 const ParticlesData& particles,
                                 const SphSimulation::Parameters& simulationData,
                                 const SphSimulation::Grid& grid,
                                 Func&& func)
{
    const auto originCell = calculateCellIndex(position, simulationData, grid);
    // Check if we're using periodic boundaries (Poiseuille flow)
    bool usePeriodic = (simulationData.testCase == cuda::Simulation::Parameters::TestCase::PoiseuilleFlow);
    const float domainWidth = simulationData.domain.max.x - simulationData.domain.min.x;
    // Standard neighbor search for surrounding cells
    for (const auto offset : offsets)
    {
        // Apply offset to get neighboring cell
        glm::uvec3 cellWithOffset = originCell + glm::uvec3 {offset.x, offset.y, offset.z};
        // For non-periodic boundaries, skip cells outside the grid
        if (!usePeriodic)
        {
            if (cellWithOffset.x >= grid.gridSize.x || cellWithOffset.y >= grid.gridSize.y ||
                cellWithOffset.z >= grid.gridSize.z || static_cast<int>(cellWithOffset.x) < 0 ||
                static_cast<int>(cellWithOffset.y) < 0 || static_cast<int>(cellWithOffset.z) < 0)
            {
                continue;
            }
        }
        else
        {
            // Check if we're wrapping in x-direction
            bool wrappingX = (cellWithOffset.x >= grid.gridSize.x || static_cast<int>(cellWithOffset.x) < 0);
            // For periodic boundaries in x-direction (Poiseuille flow)
            // Wrap around in x-direction
            if (cellWithOffset.x >= grid.gridSize.x)
            {
                cellWithOffset.x = cellWithOffset.x % grid.gridSize.x;
            }
            else if (static_cast<int>(cellWithOffset.x) < 0)
            {
                cellWithOffset.x = grid.gridSize.x - 1 - ((-static_cast<int>(cellWithOffset.x) - 1) % grid.gridSize.x);
            }
            // Standard boundary checking for y and z
            if (cellWithOffset.y >= grid.gridSize.y || cellWithOffset.z >= grid.gridSize.z ||
                static_cast<int>(cellWithOffset.y) < 0 || static_cast<int>(cellWithOffset.z) < 0)
            {
                continue;
            }
        }
        // Get the particle range for this cell
        const auto cellIdx = flattenCellIndex(cellWithOffset, grid.gridSize);
        const auto startIdx = grid.cellStartIndices.data[cellIdx];
        const auto endIdx = grid.cellEndIndices.data[cellIdx];
        if (startIdx == -1 || startIdx > endIdx)
        {
            continue;
        }
        // Process each particle in this cell
        for (auto i = startIdx; i <= endIdx; i++)
        {
            const auto neighborIdx = grid.particleArrayIndices.data[i];
            // Get actual neighbor position
            const auto neighborPos = particles.predictedPositions[neighborIdx];
            // Apply periodic boundary adjustment if needed
            glm::vec4 adjustedPos = neighborPos;
            if (usePeriodic)
            {
                float dx = neighborPos.x - position.x;

                // If the distance is greater than half the domain width,
                // it's shorter to go around the periodic boundary
                if (dx > domainWidth * 0.5f)
                {
                    // Neighbor appears to be on far right, adjust to appear from left
                    adjustedPos.x -= domainWidth;
                }
                else if (dx < -domainWidth * 0.5f)
                {
                    // Neighbor appears to be on far left, adjust to appear from right
                    adjustedPos.x += domainWidth;
                }
            }

            // Pass both the neighbor index and its adjusted position
            func(neighborIdx, adjustedPos);
        }
    }
}

}
