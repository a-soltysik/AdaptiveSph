#pragma once

#include "../SphSimulation.cuh"

namespace sph::cuda::refinement
{

struct RefinementData
{
    enum class RemovalState : uint32_t
    {
        Default = 0,
        Keep,
        Remove
    };

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
        Span<RemovalState> removalFlags;
        Span<uint32_t> prefixSums;
        uint32_t* particlesMergeCount;
    };

    SplitData split;
    MergeData merge;

    Span<uint32_t> particlesIds;
    uint32_t* particlesCount;
};

}
