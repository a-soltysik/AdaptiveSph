#pragma once

#include <cstdint>
#include <span>

#include "utils/DeviceValue.cuh"

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
        thrust::device_vector<float> criterionValues;
        thrust::device_vector<uint32_t> particlesIdsToSplit;
        DeviceValue<uint32_t> particlesSplitCount;
    };

    struct MergeData
    {
        thrust::device_vector<float> criterionValues;
        thrust::device_vector<uint32_t> eligibleParticles;
        DeviceValue<uint32_t> eligibleCount;
        thrust::device_vector<uint32_t> mergeCandidates;
        thrust::device_vector<uint32_t> mergePairs;
        thrust::device_vector<RemovalState> removalFlags;
        thrust::device_vector<uint32_t> prefixSums;
        DeviceValue<uint32_t> mergeCount;
    };

    SplitData split;
    MergeData merge;

    thrust::device_vector<uint32_t> particlesIds;
    DeviceValue<uint32_t> particlesCount;
};

struct RefinementDataView
{
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
        std::span<RefinementData::RemovalState> removalFlags;
        std::span<uint32_t> prefixSums;
        uint32_t* mergeCount;
    };

    SplitData split;
    MergeData merge;

    std::span<uint32_t> particlesIds;
    uint32_t* particlesCount;

    explicit RefinementDataView(RefinementData& data)
        : split {
            .criterionValues = {thrust::raw_pointer_cast(data.split.criterionValues.data()),
                                data.split.criterionValues.size()    },
            .particlesIdsToSplit = {thrust::raw_pointer_cast(data.split.particlesIdsToSplit.data()),
                                data.split.particlesIdsToSplit.size()},
            .particlesSplitCount = data.split.particlesSplitCount.getDevicePtr()
  },
  merge
    {
        .criterionValues = {thrust::raw_pointer_cast(data.merge.criterionValues.data()),
                            data.merge.criterionValues.size()},
        .eligibleParticles = {thrust::raw_pointer_cast(data.merge.eligibleParticles.data()),
                              data.merge.eligibleParticles.size()},
        .eligibleCount = data.merge.eligibleCount.getDevicePtr(),
        .mergeCandidates = {thrust::raw_pointer_cast(data.merge.mergeCandidates.data()), data.merge.mergeCandidates.size()},
        .mergePairs = {thrust::raw_pointer_cast(data.merge.mergePairs.data()), data.merge.mergePairs.size()},
        .removalFlags = {thrust::raw_pointer_cast(data.merge.removalFlags.data()), data.merge.removalFlags.size()},
        .prefixSums = {thrust::raw_pointer_cast(data.merge.prefixSums.data()), data.merge.prefixSums.size()},
        .mergeCount = data.merge.mergeCount.getDevicePtr()
    },
    particlesIds {thrust::raw_pointer_cast(data.particlesIds.data()), data.particlesIds.size()},
    particlesCount{data.particlesCount.getDevicePtr()}
    {
    }
};
}
