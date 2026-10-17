#pragma once

#include <cuda/simulation/Simulation.cuh>

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
                                                    Simulation::Parameters simulationData,
                                                    float lidVelocity = 0.0F);

__global__ void computeBoundaryDensityContributionPeriodic(FluidParticlesData fluidParticles,
                                                           BoundaryParticlesData boundaryParticles,
                                                           NeighborGrid::Device grid);

__global__ void computeBoundaryPressureAccelerationPeriodic(FluidParticlesData fluidParticles,
                                                            BoundaryParticlesData boundaryParticles,
                                                            NeighborGrid::Device grid,
                                                            Simulation::Parameters simulationData);

__global__ void computeBoundaryFrictionAccelerationPeriodic(FluidParticlesData fluidParticles,
                                                            BoundaryParticlesData boundaryParticles,
                                                            NeighborGrid::Device grid,
                                                            Simulation::Parameters simulationData,
                                                            float lidVelocity = 0.0F);

}
