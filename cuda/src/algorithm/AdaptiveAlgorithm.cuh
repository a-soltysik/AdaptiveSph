#pragma once

#include <cuda_runtime_api.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <cuda/simulation/Simulation.cuh>
#include <thrust/detail/sequence.inl>

#include "cuda/refinement/RefinementParameters.cuh"
#include "simulation/SphSimulation.cuh"
#include "simulation/refinement/Common.cuh"

namespace sph::cuda::refinement
{

__global__ auto splitParticles(FluidParticlesData particles,
                               RefinementDataView refinementData,
                               SplittingParameters params,
                               uint32_t maxParticleCount) -> void;

__global__ auto buildCompactionMap(RefinementDataView refinementData, uint32_t particleCount) -> void;
__global__ auto compactParticles(FluidParticlesData particles, RefinementDataView refinementData, uint32_t oldCount)
    -> void;

__global__ auto identifyMergeCandidates(FluidParticlesData particles,
                                        RefinementDataView refinementData,
                                        NeighborGrid::Device grid,
                                        Simulation::Parameters simulationData,
                                        RefinementParameters refinementParameters) -> void;

__global__ auto resolveMergePairs(RefinementDataView refinementData, uint32_t particleCount) -> void;
__global__ auto performMerges(FluidParticlesData particles,
                              RefinementDataView refinementData,
                              Simulation::Parameters simulationData) -> void;

__global__ auto updateParticleCount(RefinementData refinementData, uint32_t particleCount) -> void;

template <typename CriterionGenerator>
__global__ void getCriterionValues(FluidParticlesData particles,
                                   float* splitCriterionValues,
                                   CriterionGenerator criterionGenerator,
                                   const NeighborGrid::Device grid,
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
auto findTopParticlesToSplit(FluidParticlesData particles,
                             RefinementData& refinementData,
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
        std::min(static_cast<uint32_t>(topParticlesCount),
                 static_cast<uint32_t>(
                     std::round(static_cast<float>(particles.particleCount) * refinementParameters.maxBatchRatio)));

    refinementData.split.particlesSplitCount = maxParticlesCount;

    thrust::copy_n(thrust::device,
                   refinementData.particlesIds.data(),
                   maxParticlesCount,
                   refinementData.split.particlesIdsToSplit.data());
}

template <typename CriterionSorter>
auto findTopParticlesToMerge(FluidParticlesData particles,
                             RefinementData& refinementData,
                             RefinementParameters refinementParameters,
                             CriterionSorter criterionSorter) -> void
{
    cudaDeviceSynchronize();

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
        std::min(static_cast<uint32_t>(eligibleCount),
                 static_cast<uint32_t>(
                     std::round(static_cast<float>(particles.particleCount) * refinementParameters.maxBatchRatio)));

    thrust::copy_n(thrust::device,
                   refinementData.particlesIds.data(),
                   maxEligibleCount,
                   refinementData.merge.eligibleParticles.data());

    refinementData.merge.eligibleCount = maxEligibleCount;
}

}
