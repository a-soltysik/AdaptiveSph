#include "Iteration.hpp"
#include "cuda/Simulation.cuh"

namespace sph::cuda
{
__device__ void handlePeriodicBoundaries(const glm::ivec3& targetCell,
                                         glm::ivec3& wrappedCell,
                                         glm::vec3& positionShift,
                                         const glm::uvec3& gridSize,
                                         const glm::vec3& domainSize,
                                         Simulation::Parameters::TestCase testCase)
{
    if (testCase == Simulation::Parameters::TestCase::TaylorGreenVortex)
    {
        // Apply periodic boundaries in all directions for Taylor-Green
        applyPeriodicBoundary(targetCell.x, wrappedCell.x, positionShift.x, gridSize.x, domainSize.x);
        applyPeriodicBoundary(targetCell.y, wrappedCell.y, positionShift.y, gridSize.y, domainSize.y);
        applyPeriodicBoundary(targetCell.z, wrappedCell.z, positionShift.z, gridSize.z, domainSize.z);
    }
    else if (testCase == Simulation::Parameters::TestCase::PoiseuilleFlow)
    {
        // Apply periodic boundaries only in X direction for Poiseuille flow
        applyPeriodicBoundary(targetCell.x, wrappedCell.x, positionShift.x, gridSize.x, domainSize.x);
        // Skip cells outside bounds for Y and Z
        if (targetCell.y < 0 || targetCell.y >= static_cast<int>(gridSize.y) || targetCell.z < 0 ||
            targetCell.z >= static_cast<int>(gridSize.z))
        {
            // Indicate we should skip this cell
            wrappedCell = glm::ivec3(-1);
        }
    }
}

__device__ void applyPeriodicBoundary(
    int targetCoord, int& wrappedCoord, float& positionShift, uint32_t gridSize, float domainSize)
{
    if (targetCoord < 0)
    {
        wrappedCoord = static_cast<int>(gridSize) + (targetCoord % static_cast<int>(gridSize));
        if (wrappedCoord == static_cast<int>(gridSize))
        {
            wrappedCoord = 0;
        }
        positionShift = -domainSize;
    }
    else if (targetCoord >= static_cast<int>(gridSize))
    {
        wrappedCoord = targetCoord % static_cast<int>(gridSize);
        positionShift = domainSize;
    }
}

__device__ bool isOutsideGrid(const glm::ivec3& cell, const glm::uvec3& gridSize)
{
    return cell.x < 0 || cell.x >= static_cast<int>(gridSize.x) || cell.y < 0 ||
           cell.y >= static_cast<int>(gridSize.y) || cell.z < 0 || cell.z >= static_cast<int>(gridSize.z);
}

__device__ void applyMinimumImageConvention(const glm::vec4& position,
                                            glm::vec4& shiftedPos,
                                            const glm::vec3& domainSize,
                                            Simulation::Parameters::TestCase testCase)
{
    glm::vec3 diff = glm::vec3(shiftedPos - position);

    if (testCase == Simulation::Parameters::TestCase::TaylorGreenVortex)
    {
        // Apply minimum image convention in all dimensions
        applyMinimumImageShift(diff.x, shiftedPos.x, domainSize.x);
        applyMinimumImageShift(diff.y, shiftedPos.y, domainSize.y);
        applyMinimumImageShift(diff.z, shiftedPos.z, domainSize.z);
    }
    else if (testCase == Simulation::Parameters::TestCase::PoiseuilleFlow)
    {
        // Only apply in X dimension for Poiseuille
        applyMinimumImageShift(diff.x, shiftedPos.x, domainSize.x);
    }
}

__device__ void applyMinimumImageShift(float diff, float& coord, float domainSize)
{
    if (diff > domainSize * 0.5f)
    {
        coord -= domainSize;
    }
    else if (diff < -domainSize * 0.5f)
    {
        coord += domainSize;
    }
}
}
