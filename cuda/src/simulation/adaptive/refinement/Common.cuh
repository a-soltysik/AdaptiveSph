#pragma once

#include <cstdint>
#include <span>

namespace sph::cuda::refinement
{

struct RefinementData
{
    enum class RemovalState : uint32_t
    {
        Keep = 0,
        Remove = 1
    };

    struct SplitData
    {
        std::span<float> criterionValues;
        std::span<uint32_t> particlesIdsToSplit;
        uint32_t* particlesSplitCount;
    };

    struct MergeData
    {
        std::span<float> criterionValues;
        std::span<uint32_t> eligibleParticles;
        uint32_t* eligibleCount;
        std::span<uint32_t> mergeCandidates;
        std::span<uint32_t> mergePairs;
        std::span<RemovalState> removalFlags;
        std::span<uint32_t> prefixSums;
        uint32_t* mergeCount;
    };

    SplitData split;
    MergeData merge;

    std::span<uint32_t> particlesIds;
    uint32_t* particlesCount;
};

}
