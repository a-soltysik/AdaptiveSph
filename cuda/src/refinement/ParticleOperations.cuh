// ParticleOperations.cuh
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

struct MergeConfiguration
{
    float maxMassRatio;      // From refinement parameters
    float baseParticleMass;  // From simulation parameters
    float maxMassThreshold;  // Precomputed: maxMassRatio * baseParticleMass
};

struct MergePair
{
    uint32_t first;
    uint32_t second;
    float distance;
    bool valid;
};

struct MergeState
{
    enum class Status : uint32_t
    {
        Available = 0,
        Proposing = 1,
        Accepted = 2,
        Paired = 3
    };
    Status status;
    uint32_t partner;
    float distance;
};

struct EnhancedMergeData
{
    Span<float> criterionValues;
    Span<uint32_t> eligibleParticles;
    uint32_t* eligibleCount;

    Span<MergeState> states;
    Span<MergePair> pairs;
    uint32_t* pairCount;

    Span<uint32_t> compactionMap;
    uint32_t* newParticleCount;
};

__global__ void resolveProposals(ParticlesData particles, EnhancedMergeData mergeData);
__global__ void proposePartners(ParticlesData particles,
                                EnhancedMergeData mergeData,
                                SphSimulation::Grid grid,
                                Simulation::Parameters simulationData,
                                MergeConfiguration mergeConfig);

__global__ void markPotentialMerges(ParticlesData particles, RefinementData refinementData);

__global__ void splitParticles(ParticlesData particles,
                               RefinementData refinementData,
                               SplittingParameters params,
                               uint32_t maxParticleCount);

__global__ void mergeParticles(ParticlesData particles,
                               RefinementData refinementData,
                               Simulation::Parameters simulationData);

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

__global__ void validateMergePairs(RefinementData refinementData, uint32_t particleCount);

// Template-based getCriterionValues (no more function pointers)
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
__global__ void identifyEligibleParticles(ParticlesData particles, EnhancedMergeData mergeData, float maxMass);
__global__ void executeMerges(ParticlesData particles,
                              EnhancedMergeData mergeData,
                              Simulation::Parameters simulationData);

__global__ void createMergePairs(EnhancedMergeData mergeData);

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
                                                        return value > 0.0f;
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
                                                        return value > 0.0f;
                                                    });

    const auto maxParticlesCount =
        std::min<uint32_t>(topParticlesCount, particles.particleCount * refinementParameters.maxBatchRatio);
    cudaMemcpy(refinementData.merge.particlesMergeCount, &maxParticlesCount, sizeof(uint32_t), cudaMemcpyHostToDevice);

    thrust::copy_n(thrust::device,
                   refinementData.particlesIds.data,
                   maxParticlesCount,
                   refinementData.merge.particlesIdsToMerge.first.data);
}

__global__ void buildCompactionMap(EnhancedMergeData mergeData, uint32_t particleCount);
__global__ void compactParticles(ParticlesData particles, EnhancedMergeData mergeData, uint32_t oldCount);

}
