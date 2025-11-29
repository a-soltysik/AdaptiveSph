#include <thrust/fill.h>
#include <thrust/sort.h>

#include "../../algorithm/neighbors/NeighborKernels.cuh"
#include "NeighborGrid.cuh"

namespace sph::cuda
{
NeighborGrid::NeighborGrid(const Simulation::Parameters::Domain& domain,
                           float cellWidth,
                           uint32_t fluidParticleCapacity,
                           uint32_t boundaryParticleCapacity)
    : _domain {domain},
      _grid {createGrid(domain, cellWidth)},
      _fluid {createData(getCellCount(_grid.gridSize), fluidParticleCapacity)},
      _boundary {createData(getCellCount(_grid.gridSize), boundaryParticleCapacity)},
      _cellWidth {cellWidth}
{
}

NeighborGrid::~NeighborGrid()
{
    cudaFree(_fluid.cellEndIndices.data());
    cudaFree(_fluid.cellStartIndices.data());
    cudaFree(_fluid.particleArrayIndices.data());
    cudaFree(_fluid.particleGridIndices.data());
    cudaFree(_boundary.cellEndIndices.data());
    cudaFree(_boundary.cellStartIndices.data());
    cudaFree(_boundary.particleGridIndices.data());
    cudaFree(_boundary.particleArrayIndices.data());
}

void NeighborGrid::updateFluid(const RunData& runData, glm::vec4* positions, uint32_t fluidParticleCount)
{
    _fluid.particleCount = fluidParticleCount;
    resetGrid(_fluid);
    assignParticlesToCells(runData, _fluid, positions);
    sortParticles(_fluid);
    calculateCellStartAndEndIndices(runData, _fluid);
}

void NeighborGrid::updateBoundary(const RunData& runData, glm::vec4* positions, uint32_t boundaryParticleCount)
{
    _boundary.particleCount = boundaryParticleCount;
    resetGrid(_boundary);
    assignParticlesToCells(runData, _boundary, positions);
    sortParticles(_boundary);
    calculateCellStartAndEndIndices(runData, _boundary);
}

void NeighborGrid::updateBoundarySize(const RunData& fluidRunData,
                                      const RunData& boundaryRunData,
                                      const Simulation::Parameters::Domain& domain,
                                      glm::vec4* fluidPositions,
                                      uint32_t fluidParticleCount,
                                      glm::vec4* boundaryPositions,
                                      uint32_t boundaryParticleCount)
{
    cudaFree(_fluid.cellEndIndices.data());
    cudaFree(_fluid.cellStartIndices.data());
    cudaFree(_boundary.cellEndIndices.data());
    cudaFree(_boundary.cellStartIndices.data());

    int32_t* boundaryCellStartIndices {};
    int32_t* boundaryCellEndIndices {};
    int32_t* fluidCellStartIndices {};
    int32_t* fluidCellEndIndices {};

    _domain = domain;
    _grid = createGrid(domain, _cellWidth);

    const auto cellsCount = getCellCount(_grid.gridSize);
    cudaMalloc(&boundaryCellStartIndices, cellsCount * sizeof(int32_t));
    cudaMalloc(&boundaryCellEndIndices, cellsCount * sizeof(int32_t));
    cudaMalloc(&fluidCellStartIndices, cellsCount * sizeof(int32_t));
    cudaMalloc(&fluidCellEndIndices, cellsCount * sizeof(int32_t));

    _fluid.cellStartIndices = {fluidCellStartIndices, cellsCount};
    _fluid.cellEndIndices = {fluidCellEndIndices, cellsCount};
    _boundary.cellStartIndices = {boundaryCellStartIndices, cellsCount};
    _boundary.cellEndIndices = {boundaryCellEndIndices, cellsCount};

    updateFluid(fluidRunData, fluidPositions, fluidParticleCount);
    updateBoundary(boundaryRunData, boundaryPositions, boundaryParticleCount);
}

auto NeighborGrid::createData(uint32_t totalCells, uint32_t particleCapacity) -> GridData
{
    int32_t* particleIndices {};
    int32_t* particleArrayIndices {};
    int32_t* cellStartIndices {};
    int32_t* cellEndIndices {};

    cudaMalloc(&particleIndices, particleCapacity * sizeof(int32_t));
    cudaMalloc(&particleArrayIndices, particleCapacity * sizeof(int32_t));
    cudaMalloc(&cellStartIndices, totalCells * sizeof(int32_t));
    cudaMalloc(&cellEndIndices, totalCells * sizeof(int32_t));

    return {
        .cellStartIndices = std::span {cellStartIndices,     totalCells      },
        .cellEndIndices = std::span {cellEndIndices,       totalCells      },
        .particleGridIndices = std::span {particleIndices,      particleCapacity},
        .particleArrayIndices = std::span {particleArrayIndices, particleCapacity},
        .particleCount = 0
    };
}

auto NeighborGrid::calculateGridSize(const Simulation::Parameters::Domain& domain, float cellWidth) -> glm::ivec3
{
    return glm::ivec3 {glm::ceil(domain.getScale() / cellWidth)};
}

auto NeighborGrid::createGrid(const Simulation::Parameters::Domain& domain, float cellWidth) -> Grid
{
    const auto gridSize = calculateGridSize(domain, cellWidth);
    const auto cellSize = domain.getScale() / glm::vec3 {gridSize};
    return {
        .gridSize = gridSize,
        .cellSize = cellSize,
    };
}

auto NeighborGrid::getCellCount(const glm::ivec3& gridSize) -> uint32_t
{
    return static_cast<uint32_t>(gridSize.x * gridSize.y * gridSize.z);
}

void NeighborGrid::resetGrid(const GridData& data)
{
    thrust::fill_n(thrust::device, data.cellStartIndices.data(), data.cellStartIndices.size(), -1);
    thrust::fill_n(thrust::device, data.cellEndIndices.data(), data.cellEndIndices.size(), -1);
}

void NeighborGrid::assignParticlesToCells(const RunData& runData, const GridData& data, glm::vec4* positions)
{
    kernel::assignParticlesToCells<<<runData.blocksPerGrid, runData.threadsPerBlock>>>(positions, toDevice(), data);
}

void NeighborGrid::sortParticles(const GridData& data)
{
    thrust::sort_by_key(thrust::device,
                        data.particleGridIndices.data(),
                        data.particleGridIndices.data() + data.particleCount,
                        data.particleArrayIndices.data());
}

void NeighborGrid::calculateCellStartAndEndIndices(const RunData& runData, const GridData& data)
{
    kernel::calculateCellStartAndEndIndices<<<runData.blocksPerGrid, runData.threadsPerBlock>>>(data);
}

auto NeighborGrid::toDevice() const -> Device
{
    return {.domain = _domain, .grid = _grid, .fluid = _fluid, .boundary = _boundary};
}
}