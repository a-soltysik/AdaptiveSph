#pragma once

#include "algorithm/NeighborGrid.cuh"
#include "simulation/SphSimulation.cuh"

namespace sph::cuda::kernel
{

__global__ void assignParticlesToCells(glm::vec4* positions, NeighborGrid::Device grid, NeighborGrid::GridData data);
__global__ void calculateCellStartAndEndIndices(NeighborGrid::GridData data);

}
