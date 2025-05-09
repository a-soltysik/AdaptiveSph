#pragma once

#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <cfloat>
#include <cuda/Simulation.cuh>

#include "../SphSimulation.cuh"
#include "Common.cuh"
#include "cuda/refinement/RefinementParameters.cuh"

namespace sph::cuda::refinement
{

__global__ void splitParticles(ParticlesData particles,
                               RefinementData refinementData,
                               SplittingParameters params,
                               uint32_t maxParticleCount);

__global__ void mergeParticles(ParticlesData particles, RefinementData refinementData);

__global__ void getMergeCandidates(ParticlesData particles,
                                   RefinementData refinementData,
                                   SphSimulation::Grid grid,
                                   Simulation::Parameters simulationData);

__device__ auto getIcosahedronVertices() -> std::array<glm::vec3, 12>;

__device__ std::pair<uint32_t, float> findClosestParticle(const ParticlesData& particles,
                                                          uint32_t particleIdx,
                                                          const SphSimulation::Grid& grid,
                                                          const Simulation::Parameters& simulationData);

__global__ void removeParticles(ParticlesData particles, RefinementData refinementData);

template <typename CriterionGenerator>
__global__ void getCriterionValues(ParticlesData particles,
                                   Span<float> splitCriterionValues,
                                   CriterionGenerator criterionGenerator)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }
    const auto value = criterionGenerator(particles, idx);
    splitCriterionValues.data[idx] = value;
}

__global__ void updateParticleCount(RefinementData refinementData, uint32_t particleCount);

template <typename CriterionSorter>
void findTopParticlesToSplit(ParticlesData particles,
                             RefinementData refinementData,
                             RefinementParameters refinementParameters,
                             CriterionSorter criterionSorter)
{
    cudaDeviceSynchronize();

    thrust::sequence(thrust::device,
                     refinementData.particlesIds.data,
                     refinementData.particlesIds.data + particles.particleCount,
                     0);

    thrust::sort_by_key(thrust::device,
                        refinementData.split.criterionValues.data,
                        refinementData.split.criterionValues.data + particles.particleCount,
                        refinementData.particlesIds.data,
                        criterionSorter);

    const auto topParticlesCount = thrust::count_if(thrust::device,
                                                    refinementData.split.criterionValues.data,
                                                    refinementData.split.criterionValues.data + particles.particleCount,
                                                    [] __device__(float value) {
                                                        return value != FLT_MAX;
                                                    });

    const auto maxParticlesCount =
        std::min<uint32_t>(topParticlesCount, particles.particleCount * refinementParameters.maxBatchRatio);
    cudaMemcpy(refinementData.split.particlesSplitCount, &maxParticlesCount, sizeof(uint32_t), cudaMemcpyHostToDevice);

    thrust::copy_n(thrust::device,
                   refinementData.particlesIds.data,
                   maxParticlesCount,
                   refinementData.split.particlesIdsToSplit.data);
}

__global__ void markPotentialMerges(ParticlesData particles, RefinementData refinementData);

template <typename CriterionSorter>
void findTopParticlesToMerge(ParticlesData particles,
                             RefinementData refinementData,
                             RefinementParameters refinementParameters,
                             CriterionSorter criterionSorter)
{
    cudaDeviceSynchronize();

    thrust::sequence(thrust::device,
                     refinementData.particlesIds.data,
                     refinementData.particlesIds.data + particles.particleCount,
                     0);

    thrust::sort_by_key(thrust::device,
                        refinementData.merge.criterionValues.data,
                        refinementData.merge.criterionValues.data + particles.particleCount,
                        refinementData.particlesIds.data,
                        criterionSorter);

    const auto topParticlesCount = thrust::count_if(thrust::device,
                                                    refinementData.merge.criterionValues.data,
                                                    refinementData.merge.criterionValues.data + particles.particleCount,
                                                    [] __device__(float value) {
                                                        return value != FLT_MAX;
                                                    });

    const auto maxParticlesCount =
        std::min<uint32_t>(topParticlesCount, particles.particleCount * refinementParameters.maxBatchRatio);
    cudaMemcpy(refinementData.merge.particlesMergeCount, &maxParticlesCount, sizeof(uint32_t), cudaMemcpyHostToDevice);

    thrust::copy_n(thrust::device,
                   refinementData.particlesIds.data,
                   maxParticlesCount,
                   refinementData.merge.particlesIdsToMerge.first.data);
}

}
