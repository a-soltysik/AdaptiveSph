#pragma once
#include "../SphSimulation.cuh"
#include "Utils.cuh"

namespace sph::cuda
{

// In Iteration.hpp, update the forEachNeighbour template function:

// In common/Iteration.hpp, update the forEachNeighbour template function
template <typename Func>
__device__ void forEachNeighbour(glm::vec4 position,
                                 const ParticlesData& particles,
                                 const SphSimulation::Parameters& simulationData,
                                 const SphSimulation::Grid& grid,
                                 Func&& func)
{
    // Whether to use periodic boundaries
    bool usePeriodic = (simulationData.testCase == cuda::Simulation::Parameters::TestCase::PoiseuilleFlow ||
                        simulationData.testCase == cuda::Simulation::Parameters::TestCase::TaylorGreenVortex);

    // Domain dimensions
    const glm::vec3 domainMin = simulationData.domain.min;
    const glm::vec3 domainMax = simulationData.domain.max;
    const glm::vec3 domainSize = domainMax - domainMin;

    // Get the base cell for the current particle
    const auto centerCell = calculateCellIndex(position, simulationData, grid);

    // Process each neighboring cell (including the particle's own cell)
    for (const auto offset : offsets)
    {
        // Apply offset to get neighboring cell
        glm::ivec3 targetCell(static_cast<int>(centerCell.x) + offset.x,
                              static_cast<int>(centerCell.y) + offset.y,
                              static_cast<int>(centerCell.z) + offset.z);

        // Wrap cells for periodic boundaries
        glm::ivec3 wrappedCell = targetCell;
        glm::vec3 positionShift(0.0f);

        // Handle cell wrapping based on simulation type
        if (usePeriodic)
        {
            if (simulationData.testCase == cuda::Simulation::Parameters::TestCase::TaylorGreenVortex)
            {
                // Apply periodic boundaries in all directions for Taylor-Green

                // X direction
                if (targetCell.x < 0)
                {
                    wrappedCell.x =
                        static_cast<int>(grid.gridSize.x) + (targetCell.x % static_cast<int>(grid.gridSize.x));
                    if (wrappedCell.x == static_cast<int>(grid.gridSize.x))
                    {
                        wrappedCell.x = 0;
                    }
                    positionShift.x = -domainSize.x;
                }
                else if (targetCell.x >= static_cast<int>(grid.gridSize.x))
                {
                    wrappedCell.x = targetCell.x % static_cast<int>(grid.gridSize.x);
                    positionShift.x = domainSize.x;
                }

                // Y direction
                if (targetCell.y < 0)
                {
                    wrappedCell.y =
                        static_cast<int>(grid.gridSize.y) + (targetCell.y % static_cast<int>(grid.gridSize.y));
                    if (wrappedCell.y == static_cast<int>(grid.gridSize.y))
                    {
                        wrappedCell.y = 0;
                    }
                    positionShift.y = -domainSize.y;
                }
                else if (targetCell.y >= static_cast<int>(grid.gridSize.y))
                {
                    wrappedCell.y = targetCell.y % static_cast<int>(grid.gridSize.y);
                    positionShift.y = domainSize.y;
                }

                // Z direction
                if (targetCell.z < 0)
                {
                    wrappedCell.z =
                        static_cast<int>(grid.gridSize.z) + (targetCell.z % static_cast<int>(grid.gridSize.z));
                    if (wrappedCell.z == static_cast<int>(grid.gridSize.z))
                    {
                        wrappedCell.z = 0;
                    }
                    positionShift.z = -domainSize.z;
                }
                else if (targetCell.z >= static_cast<int>(grid.gridSize.z))
                {
                    wrappedCell.z = targetCell.z % static_cast<int>(grid.gridSize.z);
                    positionShift.z = domainSize.z;
                }
            }
            else if (simulationData.testCase == cuda::Simulation::Parameters::TestCase::PoiseuilleFlow)
            {
                // Apply periodic boundaries only in X direction for Poiseuille flow

                // X direction
                if (targetCell.x < 0)
                {
                    wrappedCell.x =
                        static_cast<int>(grid.gridSize.x) + (targetCell.x % static_cast<int>(grid.gridSize.x));
                    if (wrappedCell.x == static_cast<int>(grid.gridSize.x))
                    {
                        wrappedCell.x = 0;
                    }
                    positionShift.x = -domainSize.x;
                }
                else if (targetCell.x >= static_cast<int>(grid.gridSize.x))
                {
                    wrappedCell.x = targetCell.x % static_cast<int>(grid.gridSize.x);
                    positionShift.x = domainSize.x;
                }

                // Skip cells outside bounds for Y and Z
                if (targetCell.y < 0 || targetCell.y >= static_cast<int>(grid.gridSize.y) || targetCell.z < 0 ||
                    targetCell.z >= static_cast<int>(grid.gridSize.z))
                {
                    continue;
                }
            }
        }
        else
        {
            // For non-periodic boundaries, skip cells outside the grid
            if (targetCell.x < 0 || targetCell.x >= static_cast<int>(grid.gridSize.x) || targetCell.y < 0 ||
                targetCell.y >= static_cast<int>(grid.gridSize.y) || targetCell.z < 0 ||
                targetCell.z >= static_cast<int>(grid.gridSize.z))
            {
                continue;
            }
        }

        // Get the wrapped cell index (make sure it's within grid bounds)
        if (wrappedCell.x < 0 || wrappedCell.x >= static_cast<int>(grid.gridSize.x) || wrappedCell.y < 0 ||
            wrappedCell.y >= static_cast<int>(grid.gridSize.y) || wrappedCell.z < 0 ||
            wrappedCell.z >= static_cast<int>(grid.gridSize.z))
        {
            continue;  // Skip if still outside grid after wrapping
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

            // Get the actual neighbor position
            const auto neighborPos = particles.predictedPositions[neighborIdx];

            // Apply the position shift for periodic boundaries
            glm::vec4 shiftedPos;
            if (glm::length(positionShift) > 1e-6f)
            {
                // If we crossed a boundary, shift the particle position
                shiftedPos = neighborPos + glm::vec4(positionShift, 0.0f);
            }
            else
            {
                // Even if we didn't cross a boundary cell, we still need to apply
                // minimum image convention for the particle-particle interaction
                shiftedPos = neighborPos;

                if (usePeriodic)
                {
                    glm::vec3 diff = glm::vec3(neighborPos - position);

                    if (simulationData.testCase == cuda::Simulation::Parameters::TestCase::TaylorGreenVortex)
                    {
                        // Apply minimum image convention in all dimensions
                        if (diff.x > domainSize.x * 0.5f)
                        {
                            shiftedPos.x -= domainSize.x;
                        }
                        else if (diff.x < -domainSize.x * 0.5f)
                        {
                            shiftedPos.x += domainSize.x;
                        }

                        if (diff.y > domainSize.y * 0.5f)
                        {
                            shiftedPos.y -= domainSize.y;
                        }
                        else if (diff.y < -domainSize.y * 0.5f)
                        {
                            shiftedPos.y += domainSize.y;
                        }

                        if (diff.z > domainSize.z * 0.5f)
                        {
                            shiftedPos.z -= domainSize.z;
                        }
                        else if (diff.z < -domainSize.z * 0.5f)
                        {
                            shiftedPos.z += domainSize.z;
                        }
                    }
                    else if (simulationData.testCase == cuda::Simulation::Parameters::TestCase::PoiseuilleFlow)
                    {
                        // Only apply in X dimension for Poiseuille
                        if (diff.x > domainSize.x * 0.5f)
                        {
                            shiftedPos.x -= domainSize.x;
                        }
                        else if (diff.x < -domainSize.x * 0.5f)
                        {
                            shiftedPos.x += domainSize.x;
                        }
                    }
                }
            }

            // Pass the neighbor index and its adjusted position to the callback
            func(neighborIdx, shiftedPos);
        }
    }
}

}
