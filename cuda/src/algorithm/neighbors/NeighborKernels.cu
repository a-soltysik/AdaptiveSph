#include "NeighborKernels.cuh"

namespace sph::cuda::kernel
{
__global__ void assignParticlesToCells(glm::vec4* positions, NeighborGrid::Device grid, NeighborGrid::GridData data)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= data.particleCount)
    {
        return;
    }

    data.particleArrayIndices[idx] = idx;
    const auto cellPosition = grid.calculateCellIndex(positions[idx]);
    const auto cellIndex = grid.flattenCellIndex(cellPosition);
    data.particleGridIndices[idx] = cellIndex;
}

__global__ void calculateCellStartAndEndIndices(NeighborGrid::GridData data)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= data.particleCount)
    {
        return;
    }

    const auto cellIdx = data.particleGridIndices[idx];

    if (idx == 0 || data.particleGridIndices[idx - 1] != cellIdx)
    {
        data.cellStartIndices[cellIdx] = idx;
    }

    if (idx + 1 == data.particleCount || data.particleGridIndices[idx + 1] != cellIdx)
    {
        data.cellEndIndices[cellIdx] = idx;
    }
}

}
