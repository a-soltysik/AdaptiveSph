#pragma once

#include <vector_types.h>

#include <cstdint>
#include <glm/ext/vector_float4.hpp>
#include <span>
#include <vector>

#include "SphSimulation.cuh"
#include "cuda/Simulation.cuh"
#include "cuda/refinement/RefinementParameters.cuh"
#include "refinement/Common.cuh"
#include "utils/Memory.cuh"

namespace sph::cuda
{

class AdaptiveSphSimulation : public SphSimulation
{
public:
    AdaptiveSphSimulation(const Parameters& initialParameters,
                          const std::vector<glm::vec4>& positions,
                          const ParticlesDataBuffer& memory,
                          const refinement::RefinementParameters& refinementParams);

    void update(float deltaTime) override;

private:
    void performAdaptiveRefinement();
    void updateParticleCount();
    [[nodiscard]] auto getBlocksPerGridForParticles(uint32_t count) const -> dim3;
    void calculateMergeCriteria(std::span<float> criterionValues) const;
    void resetRefinementCounters() const;
    void identifyAndSplitParticles(uint32_t removedParticles);
    void identifyAndMergeParticles();
    void rebuildGridAfterMerge();

    Memory<float> _criterionValuesSplit;
    Memory<uint32_t> _particlesIdsToSplit;
    Memory<uint32_t> _particlesSplitCount;
    Memory<uint32_t> _particlesIds;
    Memory<uint32_t> _particlesCount;

    Memory<float> _mergeCriterionValues;
    Memory<uint32_t> _mergeEligibleParticles;
    Memory<uint32_t> _mergeEligibleCount;
    Memory<uint32_t> _mergeCandidates;
    Memory<uint32_t> _mergePairs;
    Memory<uint32_t> _mergeCount;
    Memory<refinement::RefinementData::RemovalState> _mergeRemovalFlags;
    Memory<uint32_t> _mergePrefixSums;

    refinement::RefinementParameters _refinementParams;
    refinement::RefinementData _refinementData;
    uint32_t _frameCounter = 0;
    uint32_t _targetParticleCount = 0;
};

}
