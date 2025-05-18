#include <cstdint>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/ext/vector_int3.hpp>
#include <glm/ext/vector_uint3.hpp>

#include "Iteration.cuh"
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
        applyPeriodicBoundary(targetCell.x, wrappedCell.x, positionShift.x, gridSize.x, domainSize.x);
        applyPeriodicBoundary(targetCell.y, wrappedCell.y, positionShift.y, gridSize.y, domainSize.y);
        applyPeriodicBoundary(targetCell.z, wrappedCell.z, positionShift.z, gridSize.z, domainSize.z);
    }
    else if (testCase == Simulation::Parameters::TestCase::PoiseuilleFlow)
    {
        applyPeriodicBoundary(targetCell.x, wrappedCell.x, positionShift.x, gridSize.x, domainSize.x);
        if (targetCell.y < 0 || targetCell.y >= static_cast<int32_t>(gridSize.y) || targetCell.z < 0 ||
            targetCell.z >= static_cast<int32_t>(gridSize.z))
        {
            wrappedCell = glm::ivec3(-1);
        }
    }
}

__device__ void applyPeriodicBoundary(
    int32_t targetCoord, int32_t& wrappedCoord, float& positionShift, uint32_t gridSize, float domainSize)
{
    if (targetCoord < 0)
    {
        wrappedCoord = static_cast<int32_t>(gridSize) + (targetCoord % static_cast<int32_t>(gridSize));
        if (wrappedCoord == static_cast<int32_t>(gridSize))
        {
            wrappedCoord = 0;
        }
        positionShift = -domainSize;
    }
    else if (targetCoord >= static_cast<int32_t>(gridSize))
    {
        wrappedCoord = targetCoord % static_cast<int32_t>(gridSize);
        positionShift = domainSize;
    }
}

__device__ bool isOutsideGrid(const glm::ivec3& cell, const glm::uvec3& gridSize)
{
    return cell.x < 0 || cell.x >= static_cast<int32_t>(gridSize.x) || cell.y < 0 ||
           cell.y >= static_cast<int32_t>(gridSize.y) || cell.z < 0 || cell.z >= static_cast<int32_t>(gridSize.z);
}

__device__ void adjustPositionForWrapping(const glm::vec4& position,
                                          glm::vec4& shiftedPos,
                                          const glm::vec3& domainSize,
                                          Simulation::Parameters::TestCase testCase)
{
    const auto diff = glm::vec3(shiftedPos - position);

    if (testCase == Simulation::Parameters::TestCase::TaylorGreenVortex)
    {
        // Apply minimum image convention in all dimensions
        shiftCoordinateForWrapping(diff.x, shiftedPos.x, domainSize.x);
        shiftCoordinateForWrapping(diff.y, shiftedPos.y, domainSize.y);
        shiftCoordinateForWrapping(diff.z, shiftedPos.z, domainSize.z);
    }
    else if (testCase == Simulation::Parameters::TestCase::PoiseuilleFlow)
    {
        // Only apply in X dimension for Poiseuille
        shiftCoordinateForWrapping(diff.x, shiftedPos.x, domainSize.x);
    }
}

__device__ void shiftCoordinateForWrapping(float diff, float& coord, float domainSize)
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
