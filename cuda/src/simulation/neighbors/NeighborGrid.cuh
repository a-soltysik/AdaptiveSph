#pragma once
#include <glm/vec3.hpp>
#include <span>

#include "../../../include/cuda/simulation/Simulation.cuh"

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
        std::span<int32_t> cellStartIndices;
        std::span<int32_t> cellEndIndices;
        std::span<int32_t> particleGridIndices;
        std::span<int32_t> particleArrayIndices;

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

        template <typename Func>
        __device__ void forEachFluidNeighbor(glm::vec4 position,
                                             const glm::vec4* positions,
                                             float smoothingRadius,
                                             Func&& func) const;

        template <typename Func>
        __device__ void forEachBoundaryNeighbor(glm::vec4 position,
                                                const glm::vec4* boundaryPositions,
                                                float smoothingRadius,
                                                Func&& func) const;

        __device__ auto calculateCellIndex(glm::vec4 position) const -> glm::ivec3;
        __device__ auto flattenCellIndex(glm::ivec3 cellIndex) const -> uint32_t;
    };

    NeighborGrid(const Simulation::Parameters::Domain& domain,
                 float cellWidth,
                 uint32_t fluidParticleCapacity,
                 uint32_t boundaryParticleCapacity);
    ~NeighborGrid();

    void updateFluid(const RunData& runData, glm::vec4* positions, uint32_t fluidParticleCount);
    void updateBoundary(const RunData& runData, glm::vec4* positions, uint32_t boundaryParticleCount);
    void updateBoundarySize(const RunData& fluidRunData,
                            const RunData& boundaryRunData,
                            const Simulation::Parameters::Domain& domain,
                            glm::vec4* fluidPositions,
                            uint32_t fluidParticleCount,
                            glm::vec4* boundaryPositions,
                            uint32_t boundaryParticleCount);
    [[nodiscard]] auto toDevice() const -> Device;

private:
    static auto createData(uint32_t totalCells, uint32_t particleCapacity) -> GridData;
    static auto calculateGridSize(const Simulation::Parameters::Domain& domain, float cellWidth) -> glm::ivec3;
    static auto createGrid(const Simulation::Parameters::Domain& domain, float cellWidth) -> Grid;
    static auto getCellCount(const glm::ivec3& gridSize) -> uint32_t;
    static void resetGrid(const GridData& data);
    void assignParticlesToCells(const RunData& runData, const GridData& data, glm::vec4* positions);
    void sortParticles(const GridData& data);
    void calculateCellStartAndEndIndices(const RunData& runData, const GridData& data);
    Simulation::Parameters::Domain _domain;
    Grid _grid;
    GridData _fluid;
    GridData _boundary;
    float _cellWidth;
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
                    func(neighborIdx, neighborPos);
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
                    func(boundaryIdx, boundaryPos);
                }
            }
        }
    }
}

inline __device__ auto NeighborGrid::Device::calculateCellIndex(glm::vec4 position) const -> glm::ivec3
{
    const auto relativePosition = glm::vec3 {position} - domain.min;
    const auto clampedPosition = glm::clamp(relativePosition, glm::vec3(0.F), domain.max - domain.min);

    return glm::min(glm::ivec3 {clampedPosition / grid.cellSize}, grid.gridSize - 1);
}

inline __device__ auto NeighborGrid::Device::flattenCellIndex(glm::ivec3 cellIndex) const -> uint32_t
{
    return cellIndex.x + (cellIndex.y * grid.gridSize.x) + (cellIndex.z * grid.gridSize.x * grid.gridSize.y);
}
}