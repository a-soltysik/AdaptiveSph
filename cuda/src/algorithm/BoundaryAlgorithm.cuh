#pragma once

#include <cuda/Simulation.cuh>

#include "simulation/SphSimulation.cuh"

namespace sph::cuda::kernel
{

__global__ void computeBoundaryDensityContribution(FluidParticlesData fluidParticles,
                                                   BoundaryParticlesData boundaryParticles,
                                                   NeighborGrid::Device grid);

__global__ void computeBoundaryPressureAcceleration(FluidParticlesData fluidParticles,
                                                    BoundaryParticlesData boundaryParticles,
                                                    NeighborGrid::Device grid,
                                                    Simulation::Parameters simulationData);

__global__ void computeBoundaryFrictionAcceleration(FluidParticlesData fluidParticles,
                                                    BoundaryParticlesData boundaryParticles,
                                                    NeighborGrid::Device grid,
                                                    Simulation::Parameters simulationData);

}  // namespace sph::cuda::kernel
