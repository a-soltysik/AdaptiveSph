#pragma once
#include <cuda/simulation/Simulation.cuh>
#include <glm/vec3.hpp>

#include "../../utils/DeviceBuffer.cuh"
#include "../../utils/DeviceSpan.cuh"
#include "glm/gtx/compatibility.hpp"

namespace sph::cuda
{
class NeighborGrid
{
public:
    struct Grid
    {
        glm::ivec3 gridSize;
        glm::vec3 cellSize;
    };

    struct GridData
    {
        DeviceSpan<int32_t> cellStartIndices;
        // Cell ranges use inclusive start and end indices.
        DeviceSpan<int32_t> cellEndIndices;
        DeviceSpan<int32_t> particleGridIndices;
        DeviceSpan<int32_t> particleArrayIndices;

        uint32_t particleCount;
    };

    struct RunData
    {
        dim3 blocksPerGrid;
        dim3 threadsPerBlock;
    };

    struct Device
    {
        Simulation::Parameters::Domain domain;
        Grid grid;
        GridData fluid;
        GridData boundary;
        float maxFluidSmoothingRadius;

        template <typename Func>
        __device__ void forEachFluidNeighbor(glm::vec4 position,
                                             const glm::vec4* positions,
                                             float smoothingRadius,
                                             Func&& func) const;

        template <typename Func>
        __device__ void forEachFluidNeighborPeriodic(glm::vec4 position,
                                                     const glm::vec4* positions,
                                                     float smoothingRadius,
                                                     Func&& func) const;

        template <typename Func>
        __device__ void forEachBoundaryNeighbor(glm::vec4 position,
                                                const glm::vec4* boundaryPositions,
                                                float smoothingRadius,
                                                Func&& func) const;

        template <typename Func>
        __device__ void forEachBoundaryNeighborPeriodic(glm::vec4 position,
                                                        const glm::vec4* boundaryPositions,
                                                        float smoothingRadius,
                                                        Func&& func) const;

        __device__ auto getFluidNeighborSearchRadius(float smoothingRadius, float supportRadiusRatio) const -> float;

        __device__ auto calculateCellIndex(glm::vec4 position) const -> glm::ivec3;
        __device__ auto flattenCellIndex(glm::ivec3 cellIndex) const -> uint32_t;
    };

    NeighborGrid(const Simulation::Parameters::Domain& domain,
                 float cellWidth,
                 uint32_t fluidParticleCapacity,
                 uint32_t boundaryParticleCapacity);
    NeighborGrid(const NeighborGrid&) = delete;
    auto operator=(const NeighborGrid&) -> NeighborGrid& = delete;
    NeighborGrid(NeighborGrid&&) = delete;
    auto operator=(NeighborGrid&&) -> NeighborGrid& = delete;
    ~NeighborGrid();

    void updateFluid(const RunData& runData,
                     glm::vec4* positions,
                     const float* smoothingRadii,
                     uint32_t fluidParticleCount);
    void updateBoundary(const RunData& runData, glm::vec4* positions, uint32_t boundaryParticleCount);
    void updateBoundarySize(const RunData& fluidRunData,
                            const RunData& boundaryRunData,
                            const Simulation::Parameters::Domain& domain,
                            glm::vec4* fluidPositions,
                            const float* fluidSmoothingRadii,
                            uint32_t fluidParticleCount,
                            glm::vec4* boundaryPositions,
                            uint32_t boundaryParticleCount);
    [[nodiscard]] auto toDevice() -> Device;

private:
    struct GridStorage
    {
        DeviceBuffer<int32_t> cellStartIndices;
        DeviceBuffer<int32_t> cellEndIndices;
        DeviceBuffer<int32_t> particleGridIndices;
        DeviceBuffer<int32_t> particleArrayIndices;
        uint32_t particleCount {};
        uint32_t particleCapacity {};

        [[nodiscard]] auto view() -> GridData
        {
            return {.cellStartIndices = cellStartIndices.view(),
                    .cellEndIndices = cellEndIndices.view(),
                    .particleGridIndices = particleGridIndices.view(),
                    .particleArrayIndices = particleArrayIndices.view(),
                    .particleCount = particleCount};
        }
    };

    static auto createData(uint32_t totalCells, uint32_t particleCapacity, const char* context) -> GridStorage;
    static auto calculateGridSize(const Simulation::Parameters::Domain& domain, float cellWidth) -> glm::ivec3;
    static auto createGrid(const Simulation::Parameters::Domain& domain, float cellWidth) -> Grid;
    static auto getCellCount(const glm::ivec3& gridSize) -> uint32_t;
    static void resetGrid(const GridData& data);
    void assignParticlesToCells(const RunData& runData, const GridData& data, glm::vec4* positions);
    void sortParticles(const GridData& data);
    void calculateCellStartAndEndIndices(const RunData& runData, const GridData& data);
    Simulation::Parameters::Domain _domain;
    Grid _grid;
    GridStorage _fluid;
    GridStorage _boundary;
    float _cellWidth;
    float _maxFluidSmoothingRadius {};
};

template <typename Func>
void __device__ NeighborGrid::Device::forEachFluidNeighbor(glm::vec4 position,
                                                           const glm::vec4* positions,
                                                           float smoothingRadius,
                                                           Func&& func) const
{
    const auto min = glm::max(glm::ivec3 {(glm::vec3 {position} - domain.min - smoothingRadius) / grid.cellSize},
                              glm::ivec3 {0, 0, 0});

    const auto max =
        glm::min(glm::ivec3 {(glm::vec3 {position} - domain.min + smoothingRadius) / grid.cellSize}, grid.gridSize - 1);

    for (auto x = min.x; x <= max.x; x++)
    {
        for (auto y = min.y; y <= max.y; y++)
        {
            for (auto z = min.z; z <= max.z; z++)
            {
                const auto cellIdx = flattenCellIndex(glm::ivec3 {x, y, z});
                const auto startIdx = fluid.cellStartIndices[cellIdx];
                const auto endIdx = fluid.cellEndIndices[cellIdx];

                if (startIdx == -1 || startIdx > endIdx)
                {
                    continue;
                }

                for (auto i = startIdx; i <= endIdx; i++)
                {
                    const auto neighborIdx = fluid.particleArrayIndices[i];
                    const auto neighborPos = positions[neighborIdx];
                    if (glm::isfinite(neighborPos.x) && glm::isfinite(neighborPos.y) && glm::isfinite(neighborPos.z))
                    {
                        func(neighborIdx, neighborPos);
                    }
                }
            }
        }
    }
}

template <typename Func>
void __device__ NeighborGrid::Device::forEachFluidNeighborPeriodic(glm::vec4 position,
                                                                   const glm::vec4* positions,
                                                                   float smoothingRadius,
                                                                   Func&& func) const
{
    const auto min = glm::max(glm::ivec3 {(glm::vec3 {position} - domain.min - smoothingRadius) / grid.cellSize},
                              glm::ivec3 {0, 0, 0});

    const auto max =
        glm::min(glm::ivec3 {(glm::vec3 {position} - domain.min + smoothingRadius) / grid.cellSize}, grid.gridSize - 1);

    for (auto x = min.x; x <= max.x; x++)
    {
        for (auto y = min.y; y <= max.y; y++)
        {
            for (auto z = min.z; z <= max.z; z++)
            {
                const auto cellIdx = flattenCellIndex(glm::ivec3 {x, y, z});
                const auto startIdx = fluid.cellStartIndices[cellIdx];
                const auto endIdx = fluid.cellEndIndices[cellIdx];

                if (startIdx == -1 || startIdx > endIdx)
                {
                    continue;
                }

                for (auto i = startIdx; i <= endIdx; i++)
                {
                    const auto neighborIdx = fluid.particleArrayIndices[i];
                    const auto neighborPos = positions[neighborIdx];
                    if (glm::isfinite(neighborPos.x) && glm::isfinite(neighborPos.y) && glm::isfinite(neighborPos.z))
                    {
                        func(neighborIdx, neighborPos);
                    }
                }
            }
        }
    }

    const auto xOutsideLeft =
        static_cast<int32_t>(std::floor((position.x - domain.min.x - smoothingRadius) / grid.cellSize.x));
    const auto xOutsideRight =
        static_cast<int32_t>(std::floor((position.x - domain.min.x + smoothingRadius) / grid.cellSize.x));
    const auto domainWidth = domain.max.x - domain.min.x;

    if (xOutsideLeft < 0)
    {
        for (auto x = grid.gridSize.x + xOutsideLeft; x <= grid.gridSize.x - 1; x++)
        {
            for (auto y = min.y; y <= max.y; y++)
            {
                for (auto z = min.z; z <= max.z; z++)
                {
                    const auto cellIdx = flattenCellIndex(glm::ivec3 {x, y, z});
                    const auto startIdx = fluid.cellStartIndices[cellIdx];
                    const auto endIdx = fluid.cellEndIndices[cellIdx];

                    if (startIdx == -1 || startIdx > endIdx)
                    {
                        continue;
                    }

                    for (auto i = startIdx; i <= endIdx; i++)
                    {
                        const auto neighborIdx = fluid.particleArrayIndices[i];
                        const auto neighborPos = positions[neighborIdx];
                        if (glm::isfinite(neighborPos.x) && glm::isfinite(neighborPos.y) &&
                            glm::isfinite(neighborPos.z))
                        {
                            func(neighborIdx, neighborPos - glm::vec4 {domainWidth, 0, 0, 0});
                        }
                    }
                }
            }
        }
    }

    if (xOutsideRight >= grid.gridSize.x)
    {
        for (auto x = 0; x <= xOutsideRight - grid.gridSize.x; x++)
        {
            for (auto y = min.y; y <= max.y; y++)
            {
                for (auto z = min.z; z <= max.z; z++)
                {
                    const auto cellIdx = flattenCellIndex(glm::ivec3 {x, y, z});
                    const auto startIdx = fluid.cellStartIndices[cellIdx];
                    const auto endIdx = fluid.cellEndIndices[cellIdx];

                    if (startIdx == -1 || startIdx > endIdx)
                    {
                        continue;
                    }

                    for (auto i = startIdx; i <= endIdx; i++)
                    {
                        const auto neighborIdx = fluid.particleArrayIndices[i];
                        const auto neighborPos = positions[neighborIdx];
                        if (glm::isfinite(neighborPos.x) && glm::isfinite(neighborPos.y) &&
                            glm::isfinite(neighborPos.z))
                        {
                            func(neighborIdx, neighborPos + glm::vec4 {domainWidth, 0, 0, 0});
                        }
                    }
                }
            }
        }
    }
}

template <typename Func>
void __device__ NeighborGrid::Device::forEachBoundaryNeighbor(glm::vec4 position,
                                                              const glm::vec4* boundaryPositions,
                                                              float smoothingRadius,
                                                              Func&& func) const
{
    const auto min = glm::max(glm::ivec3 {(glm::vec3 {position} - domain.min - smoothingRadius) / grid.cellSize},
                              glm::ivec3 {0, 0, 0});

    const auto max =
        glm::min(glm::ivec3 {(glm::vec3 {position} - domain.min + smoothingRadius) / grid.cellSize}, grid.gridSize - 1);

    for (auto x = min.x; x <= max.x; x++)
    {
        for (auto y = min.y; y <= max.y; y++)
        {
            for (auto z = min.z; z <= max.z; z++)
            {
                const auto cellIdx = flattenCellIndex(glm::ivec3 {x, y, z});
                const auto startIdx = boundary.cellStartIndices[cellIdx];
                const auto endIdx = boundary.cellEndIndices[cellIdx];

                if (startIdx == -1 || startIdx > endIdx)
                {
                    continue;
                }

                for (auto i = startIdx; i <= endIdx; i++)
                {
                    const auto boundaryIdx = boundary.particleArrayIndices[i];
                    const auto boundaryPos = boundaryPositions[boundaryIdx];
                    if (glm::isfinite(boundaryPos.x) && glm::isfinite(boundaryPos.y) && glm::isfinite(boundaryPos.z))
                    {
                        func(boundaryIdx, boundaryPos);
                    }
                }
            }
        }
    }
}

template <typename Func>
void __device__ NeighborGrid::Device::forEachBoundaryNeighborPeriodic(glm::vec4 position,
                                                                      const glm::vec4* boundaryPositions,
                                                                      float smoothingRadius,
                                                                      Func&& func) const
{
    const auto min = glm::max(glm::ivec3 {(glm::vec3 {position} - domain.min - smoothingRadius) / grid.cellSize},
                              glm::ivec3 {0, 0, 0});

    const auto max =
        glm::min(glm::ivec3 {(glm::vec3 {position} - domain.min + smoothingRadius) / grid.cellSize}, grid.gridSize - 1);

    for (auto x = min.x; x <= max.x; x++)
    {
        for (auto y = min.y; y <= max.y; y++)
        {
            for (auto z = min.z; z <= max.z; z++)
            {
                const auto cellIdx = flattenCellIndex(glm::ivec3 {x, y, z});
                const auto startIdx = boundary.cellStartIndices[cellIdx];
                const auto endIdx = boundary.cellEndIndices[cellIdx];

                if (startIdx == -1 || startIdx > endIdx)
                {
                    continue;
                }

                for (auto i = startIdx; i <= endIdx; i++)
                {
                    const auto boundaryIdx = boundary.particleArrayIndices[i];
                    const auto boundaryPos = boundaryPositions[boundaryIdx];
                    if (glm::isfinite(boundaryPos.x) && glm::isfinite(boundaryPos.y) && glm::isfinite(boundaryPos.z))
                    {
                        func(boundaryIdx, boundaryPos);
                    }
                }
            }
        }
    }

    const auto domainWidth = domain.max.x - domain.min.x;
    const auto xOutsideLeft =
        static_cast<int32_t>(std::floor((position.x - domain.min.x - smoothingRadius) / grid.cellSize.x));
    const auto xOutsideRight =
        static_cast<int32_t>(std::floor((position.x - domain.min.x + smoothingRadius) / grid.cellSize.x));

    if (xOutsideLeft < 0)
    {
        for (auto x = grid.gridSize.x + xOutsideLeft; x <= grid.gridSize.x - 1; x++)
        {
            for (auto y = min.y; y <= max.y; y++)
            {
                for (auto z = min.z; z <= max.z; z++)
                {
                    const auto cellIdx = flattenCellIndex(glm::ivec3 {x, y, z});
                    const auto startIdx = boundary.cellStartIndices[cellIdx];
                    const auto endIdx = boundary.cellEndIndices[cellIdx];

                    if (startIdx == -1 || startIdx > endIdx)
                    {
                        continue;
                    }

                    for (auto i = startIdx; i <= endIdx; i++)
                    {
                        const auto boundaryIdx = boundary.particleArrayIndices[i];
                        const auto boundaryPos = boundaryPositions[boundaryIdx];
                        if (glm::isfinite(boundaryPos.x) && glm::isfinite(boundaryPos.y) &&
                            glm::isfinite(boundaryPos.z))
                        {
                            func(boundaryIdx, boundaryPos - glm::vec4 {domainWidth, 0, 0, 0});
                        }
                    }
                }
            }
        }
    }

    if (xOutsideRight >= grid.gridSize.x)
    {
        for (auto x = 0; x <= xOutsideRight - grid.gridSize.x; x++)
        {
            for (auto y = min.y; y <= max.y; y++)
            {
                for (auto z = min.z; z <= max.z; z++)
                {
                    const auto cellIdx = flattenCellIndex(glm::ivec3 {x, y, z});
                    const auto startIdx = boundary.cellStartIndices[cellIdx];
                    const auto endIdx = boundary.cellEndIndices[cellIdx];

                    if (startIdx == -1 || startIdx > endIdx)
                    {
                        continue;
                    }

                    for (auto i = startIdx; i <= endIdx; i++)
                    {
                        const auto boundaryIdx = boundary.particleArrayIndices[i];
                        const auto boundaryPos = boundaryPositions[boundaryIdx];
                        // Neighbors on the left side of domain, shift them right
                        if (glm::isfinite(boundaryPos.x) && glm::isfinite(boundaryPos.y) &&
                            glm::isfinite(boundaryPos.z))
                        {
                            func(boundaryIdx, boundaryPos + glm::vec4 {domainWidth, 0, 0, 0});
                        }
                    }
                }
            }
        }
    }
}

inline __device__ auto NeighborGrid::Device::getFluidNeighborSearchRadius(float smoothingRadius,
                                                                          float supportRadiusRatio) const -> float
{
    return supportRadiusRatio * glm::max(smoothingRadius, maxFluidSmoothingRadius);
}

inline __device__ auto NeighborGrid::Device::calculateCellIndex(glm::vec4 position) const -> glm::ivec3
{
    const auto relativePosition = glm::vec3 {position} - domain.min;
    const auto cellIndex = glm::ivec3 {relativePosition / grid.cellSize};

    // Clamp to valid grid range [0, gridSize-1]
    return glm::clamp(cellIndex, glm::ivec3 {0}, grid.gridSize - 1);
}

inline __device__ auto NeighborGrid::Device::flattenCellIndex(glm::ivec3 cellIndex) const -> uint32_t
{
    return cellIndex.x + (cellIndex.y * grid.gridSize.x) + (cellIndex.z * grid.gridSize.x * grid.gridSize.y);
}
}
