#include "NeighborKernels.cuh"

namespace sph::cuda::kernel
{
__global__ void assignParticlesToCells(glm::vec4* positions, NeighborGrid::Device grid, NeighborGrid::GridData data)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < data.particleCount)
    {
        data.particleArrayIndices[idx] = idx;
        const auto cellPosition = grid.calculateCellIndex(positions[idx]);
        const auto cellIndex = grid.flattenCellIndex(cellPosition);
        data.particleGridIndices[idx] = cellIndex;
    }
}

__global__ void calculateCellStartAndEndIndices(NeighborGrid::GridData data)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= data.particleCount)
    {
        return;
    }

    const auto cellIdx = data.particleGridIndices[idx];

    if (idx == data.particleCount - 1)
    {
        data.cellEndIndices[cellIdx] = idx;
        return;
    }
    if (idx == 0)
    {
        data.cellStartIndices[cellIdx] = idx;
        return;
    }

    const auto cellIdxNext = data.particleGridIndices[idx + 1];
    if (cellIdx != cellIdxNext)
    {
        data.cellStartIndices[cellIdxNext] = idx + 1;
        data.cellEndIndices[cellIdx] = idx;
    }
}

}