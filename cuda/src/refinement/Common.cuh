#pragma once

#include <cuda/Simulation.cuh>

#include "../SphSimulation.cuh"

namespace sph::cuda::refinement
{

__device__ float findClosestParticle(const ParticlesData& particles,
                                     uint32_t particleIdx,
                                     const SphSimulation::State& state,
                                     const Simulation::Parameters& simulationData,
                                     uint32_t* closestIdx);

struct RefinementData
{
    struct SplitData
    {
        Span<float> criterionValues;
        Span<uint32_t> particlesIdsToSplit;
        uint32_t* particlesSplitCount;
    };

    struct MergeData
    {
        Span<float> criterionValues;
        std::pair<Span<uint32_t>, Span<uint32_t>> particlesIdsToMerge;
        Span<uint32_t> removalFlags;
        Span<uint32_t> prefixSums;
        uint32_t* particlesMergeCount;
    };

    SplitData split;
    MergeData merge;

    Span<uint32_t> particlesIds;
    uint32_t* particlesCount;
};

}  // namespace sph::cuda::refinement
