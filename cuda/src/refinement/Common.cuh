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
        std::span<float> criterionValues;       // Sorting criterion for each particle
        std::span<uint32_t> eligibleParticles;  // Particles eligible for merging
        uint32_t* eligibleCount;                // Count of eligible particles
        std::span<uint32_t> mergeCandidates;    // Each particle's best merge candidate
        std::span<uint32_t> mergePairs;         // Final merge pairs (2 per pair)
        std::span<RemovalState> removalFlags;   // Marks particles for removal
        std::span<uint32_t> prefixSums;         // For compaction after merging
        uint32_t* mergeCount;                   // Count of merged pairs
    };

    SplitData split;
    MergeData merge;

    std::span<uint32_t> particlesIds;
    uint32_t* particlesCount;
};

}
