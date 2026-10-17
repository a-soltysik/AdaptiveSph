#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/sort.h>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "../../algorithm/neighbors/NeighborKernels.cuh"
#include "../../utils/CudaError.cuh"
#include "NeighborGrid.cuh"

namespace sph::cuda
{
NeighborGrid::NeighborGrid(const Simulation::Parameters::Domain& domain,
                           float cellWidth,
                           uint32_t fluidParticleCapacity,
                           uint32_t boundaryParticleCapacity)
    : _domain {domain},
      _grid {createGrid(domain, cellWidth)},
      _fluid {createData(getCellCount(_grid.gridSize), fluidParticleCapacity, "NeighborGrid fluid storage")},
      _boundary {createData(getCellCount(_grid.gridSize), boundaryParticleCapacity, "NeighborGrid boundary storage")},
      _cellWidth {cellWidth}
{
}

NeighborGrid::~NeighborGrid() = default;

void NeighborGrid::updateFluid(const RunData& runData,
                               glm::vec4* positions,
                               const float* smoothingRadii,
                               uint32_t fluidParticleCount)
{
    if (fluidParticleCount != 0 && (positions == nullptr || smoothingRadii == nullptr))
    {
        throw std::invalid_argument {
            "NeighborGrid::updateFluid requires particle and smoothing-radius device pointers"};
    }
    if (fluidParticleCount > _fluid.particleCapacity)
    {
        throw std::length_error {"NeighborGrid::updateFluid particle count " + std::to_string(fluidParticleCount) +
                                 " exceeds capacity " + std::to_string(_fluid.particleCapacity)};
    }

    _fluid.particleCount = fluidParticleCount;
    const auto data = _fluid.view();
    resetGrid(data);
    if (fluidParticleCount == 0)
    {
        _maxFluidSmoothingRadius = 0.F;
        return;
    }

    const auto maxSmoothingRadius =
        thrust::max_element(thrust::device, smoothingRadii, smoothingRadii + fluidParticleCount);
    detail::checkCuda(cudaMemcpy(&_maxFluidSmoothingRadius, maxSmoothingRadius, sizeof(float), cudaMemcpyDeviceToHost),
                      "NeighborGrid::updateFluid: copy maximum smoothing radius to host");
    assignParticlesToCells(runData, data, positions);
    sortParticles(data);
    calculateCellStartAndEndIndices(runData, data);
}

void NeighborGrid::updateBoundary(const RunData& runData, glm::vec4* positions, uint32_t boundaryParticleCount)
{
    if (boundaryParticleCount != 0 && positions == nullptr)
    {
        throw std::invalid_argument {"NeighborGrid::updateBoundary requires a boundary-position device pointer"};
    }
    if (boundaryParticleCount > _boundary.particleCapacity)
    {
        throw std::length_error {"NeighborGrid::updateBoundary particle count " +
                                 std::to_string(boundaryParticleCount) + " exceeds capacity " +
                                 std::to_string(_boundary.particleCapacity)};
    }

    _boundary.particleCount = boundaryParticleCount;
    const auto data = _boundary.view();
    resetGrid(data);
    if (boundaryParticleCount == 0)
    {
        return;
    }

    assignParticlesToCells(runData, data, positions);
    sortParticles(data);
    calculateCellStartAndEndIndices(runData, data);
}

void NeighborGrid::updateBoundarySize(const RunData& fluidRunData,
                                      const RunData& boundaryRunData,
                                      const Simulation::Parameters::Domain& domain,
                                      glm::vec4* fluidPositions,
                                      const float* fluidSmoothingRadii,
                                      uint32_t fluidParticleCount,
                                      glm::vec4* boundaryPositions,
                                      uint32_t boundaryParticleCount)
{
    if (fluidParticleCount > _fluid.particleCapacity || boundaryParticleCount > _boundary.particleCapacity)
    {
        throw std::length_error {"NeighborGrid::updateBoundarySize particle count exceeds the allocated grid capacity"};
    }

    const auto newGrid = createGrid(domain, _cellWidth);
    const auto cellsCount = getCellCount(newGrid.gridSize);
    auto fluidCellStartIndices = DeviceBuffer<int32_t> {cellsCount, "NeighborGrid: allocate fluid cell-start indices"};
    auto fluidCellEndIndices = DeviceBuffer<int32_t> {cellsCount, "NeighborGrid: allocate fluid cell-end indices"};
    auto boundaryCellStartIndices =
        DeviceBuffer<int32_t> {cellsCount, "NeighborGrid: allocate boundary cell-start indices"};
    auto boundaryCellEndIndices =
        DeviceBuffer<int32_t> {cellsCount, "NeighborGrid: allocate boundary cell-end indices"};

    _fluid.cellStartIndices = std::move(fluidCellStartIndices);
    _fluid.cellEndIndices = std::move(fluidCellEndIndices);
    _boundary.cellStartIndices = std::move(boundaryCellStartIndices);
    _boundary.cellEndIndices = std::move(boundaryCellEndIndices);
    _domain = domain;
    _grid = newGrid;

    updateFluid(fluidRunData, fluidPositions, fluidSmoothingRadii, fluidParticleCount);
    updateBoundary(boundaryRunData, boundaryPositions, boundaryParticleCount);
}

auto NeighborGrid::createData(uint32_t totalCells, uint32_t particleCapacity, const char* context) -> GridStorage
{
    if (particleCapacity > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))
    {
        throw std::length_error {std::string {context} + " particle capacity exceeds the signed grid-index range"};
    }

    const auto cellStartContext = std::string {context} + ": allocate cell-start indices";
    const auto cellEndContext = std::string {context} + ": allocate cell-end indices";
    const auto particleGridContext = std::string {context} + ": allocate particle grid indices";
    const auto particleArrayContext = std::string {context} + ": allocate particle array indices";

    return {
        .cellStartIndices = DeviceBuffer<int32_t> {totalCells,       cellStartContext    },
        .cellEndIndices = DeviceBuffer<int32_t> {totalCells,       cellEndContext      },
        .particleGridIndices = DeviceBuffer<int32_t> {particleCapacity, particleGridContext },
        .particleArrayIndices = DeviceBuffer<int32_t> {particleCapacity, particleArrayContext},
        .particleCount = 0,
        .particleCapacity = particleCapacity
    };
}

auto NeighborGrid::calculateGridSize(const Simulation::Parameters::Domain& domain, float cellWidth) -> glm::ivec3
{
    if (!std::isfinite(cellWidth) || cellWidth <= 0.F)
    {
        throw std::invalid_argument {"NeighborGrid cell width must be finite and greater than zero"};
    }

    const auto scale = domain.getScale();
    const auto getAxisCellCount = [cellWidth](float axisScale) {
        if (!std::isfinite(axisScale) || axisScale <= 0.F)
        {
            throw std::invalid_argument {"NeighborGrid domain dimensions must be finite and greater than zero"};
        }

        const auto cellCount = std::ceil(static_cast<double>(axisScale) / static_cast<double>(cellWidth));
        if (cellCount > static_cast<double>(std::numeric_limits<int32_t>::max()))
        {
            throw std::length_error {"NeighborGrid axis cell count exceeds the supported range"};
        }
        return static_cast<int32_t>(cellCount);
    };

    return {getAxisCellCount(scale.x), getAxisCellCount(scale.y), getAxisCellCount(scale.z)};
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
    auto cellCount = uint64_t {1};
    for (const auto axisCellCount : std::array {gridSize.x, gridSize.y, gridSize.z})
    {
        if (axisCellCount <= 0)
        {
            throw std::invalid_argument {"NeighborGrid axis cell count must be greater than zero"};
        }

        const auto axisSize = static_cast<uint64_t>(axisCellCount);
        if (cellCount > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) / axisSize)
        {
            throw std::length_error {"NeighborGrid cell count exceeds the signed grid-index range"};
        }
        cellCount *= axisSize;
    }
    return static_cast<uint32_t>(cellCount);
}

void NeighborGrid::resetGrid(const GridData& data)
{
    thrust::fill_n(thrust::device, data.cellStartIndices.data(), data.cellStartIndices.size(), -1);
    thrust::fill_n(thrust::device, data.cellEndIndices.data(), data.cellEndIndices.size(), -1);
}

void NeighborGrid::assignParticlesToCells(const RunData& runData, const GridData& data, glm::vec4* positions)
{
    kernel::assignParticlesToCells<<<runData.blocksPerGrid, runData.threadsPerBlock>>>(positions, toDevice(), data);
    detail::checkCuda(cudaGetLastError(), "NeighborGrid::assignParticlesToCells: launch kernel");
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
    detail::checkCuda(cudaGetLastError(), "NeighborGrid::calculateCellStartAndEndIndices: launch kernel");
}

auto NeighborGrid::toDevice() -> Device
{
    return {.domain = _domain,
            .grid = _grid,
            .fluid = _fluid.view(),
            .boundary = _boundary.view(),
            .maxFluidSmoothingRadius = _maxFluidSmoothingRadius};
}
}
