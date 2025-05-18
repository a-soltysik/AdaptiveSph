#pragma once

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <cstdint>
#include <cuda/Simulation.cuh>
#include <span>
#include <thrust/detail/sequence.inl>

#include "../SphSimulation.cuh"
#include "Common.cuh"
#include "cuda/refinement/RefinementParameters.cuh"

namespace sph::cuda::refinement
{

__global__ auto splitParticles(ParticlesData particles,
                               RefinementData refinementData,
                               SplittingParameters params,
                               uint32_t maxParticleCount) -> void;

__global__ auto buildCompactionMap(RefinementData::MergeData mergeData, uint32_t particleCount) -> void;
__global__ auto compactParticles(ParticlesData particles, RefinementData::MergeData mergeData, uint32_t oldCount)
    -> void;

__global__ auto identifyMergeCandidates(ParticlesData particles,
                                        RefinementData::MergeData mergeData,
                                        SphSimulation::Grid grid,
                                        Simulation::Parameters simulationData,
                                        RefinementParameters refinementParameters) -> void;

__global__ auto resolveMergePairs(RefinementData::MergeData mergeData, uint32_t particleCount) -> void;
__global__ auto performMerges(ParticlesData particles,
                              RefinementData::MergeData mergeData,
                              Simulation::Parameters simulationData) -> void;

__global__ auto updateParticleCount(RefinementData refinementData, uint32_t particleCount) -> void;

template <typename CriterionGenerator>
__global__ void getCriterionValues(ParticlesData particles,
                                   std::span<float> splitCriterionValues,
                                   CriterionGenerator criterionGenerator,
                                   const SphSimulation::Grid grid,
                                   const Simulation::Parameters simulationData)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }
    const auto value = criterionGenerator(particles, idx, grid, simulationData);
    splitCriterionValues[idx] = value;
}

template <typename CriterionSorter>
auto findTopParticlesToSplit(ParticlesData particles,
                             RefinementData refinementData,
                             RefinementParameters refinementParameters,
                             CriterionSorter criterionSorter) -> void
{
    cudaDeviceSynchronize();

    thrust::sequence(thrust::device,
                     refinementData.particlesIds.data(),
                     refinementData.particlesIds.data() + particles.particleCount,
                     0);

    thrust::sort_by_key(thrust::device,
                        refinementData.split.criterionValues.data(),
                        refinementData.split.criterionValues.data() + particles.particleCount,
                        refinementData.particlesIds.data(),
                        criterionSorter);

    const auto topParticlesCount =
        thrust::count_if(thrust::device,
                         refinementData.split.criterionValues.data(),
                         refinementData.split.criterionValues.data() + particles.particleCount,
                         [] __device__(float value) {
                             return value > 0.0F;
                         });

    const auto maxParticlesCount =
        std::min<uint32_t>(topParticlesCount, particles.particleCount * refinementParameters.maxBatchRatio);
    cudaMemcpy(refinementData.split.particlesSplitCount, &maxParticlesCount, sizeof(uint32_t), cudaMemcpyHostToDevice);

    thrust::copy_n(thrust::device,
                   refinementData.particlesIds.data(),
                   maxParticlesCount,
                   refinementData.split.particlesIdsToSplit.data());
}

template <typename CriterionSorter>
auto findTopParticlesToMerge(ParticlesData particles,
                             RefinementData refinementData,
                             RefinementParameters refinementParameters,
                             CriterionSorter criterionSorter) -> void
{
    cudaDeviceSynchronize();

    // Sort particles by criterion value
    thrust::sequence(thrust::device,
                     refinementData.particlesIds.data(),
                     refinementData.particlesIds.data() + particles.particleCount,
                     0);

    thrust::sort_by_key(thrust::device,
                        refinementData.merge.criterionValues.data(),
                        refinementData.merge.criterionValues.data() + particles.particleCount,
                        refinementData.particlesIds.data(),
                        criterionSorter);
    const auto eligibleCount = thrust::count_if(thrust::device,
                                                refinementData.merge.criterionValues.data(),
                                                refinementData.merge.criterionValues.data() + particles.particleCount,
                                                [] __device__(float value) {
                                                    return value > 0.0F;
                                                });
    const auto maxEligibleCount =
        std::min<uint32_t>(eligibleCount, particles.particleCount * refinementParameters.maxBatchRatio);

    thrust::copy_n(thrust::device,
                   refinementData.particlesIds.data(),
                   maxEligibleCount,
                   refinementData.merge.eligibleParticles.data());

    cudaMemcpy(refinementData.merge.eligibleCount, &maxEligibleCount, sizeof(uint32_t), cudaMemcpyHostToDevice);
}

}
