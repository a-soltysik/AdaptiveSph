#pragma once
#include "cuda/Simulation.cuh"
#include "simulation/SphSimulation.cuh"
#include "utils/NeighborSharingUtils.cuh"

namespace sph::cuda::kernel
{
__global__ void computeParticleRectangles(ParticlesData particles,
                                          Rectangle3D* rectangles,
                                          SphSimulation::Grid grid,
                                          Simulation::Parameters simulationData);

__global__ void computeDensitiesENS(ParticlesData particles,
                                    const Rectangle3D* particleRectangles,
                                    SphSimulation::Grid grid,
                                    Simulation::Parameters simulationData,
                                    int32_t particlesPerBatch);

__global__ void computePressureForceENS(ParticlesData particles,
                                        const Rectangle3D* particleRectangles,
                                        SphSimulation::Grid grid,
                                        Simulation::Parameters simulationData,
                                        float dt,
                                        int32_t particlesPerBatch);

__global__ void computeViscosityForceENS(ParticlesData particles,
                                         const Rectangle3D* particleRectangles,
                                         SphSimulation::Grid grid,
                                         Simulation::Parameters simulationData,
                                         float dt,
                                         int32_t particlesPerBatch);

}