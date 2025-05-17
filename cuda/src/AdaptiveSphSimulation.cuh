#pragma once

#include <vector_types.h>

#include <cstdint>
#include <glm/ext/vector_float4.hpp>
#include <vector>

#include "SphSimulation.cuh"
#include "common/Memory.cuh"
#include "cuda/Simulation.cuh"
#include "cuda/refinement/RefinementParameters.cuh"
#include "refinement/Common.cuh"
#include "refinement/ParticleOperations.cuh"

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
    void identifyAndSplitParticles() const;
    void identifyAndMergeParticles() const;
    void computePrefixSum() const;

private:
    void performAdaptiveRefinement();
    void updateParticleCount();
    [[nodiscard]] auto getBlocksPerGridForParticles(uint32_t count) const -> dim3;
    void performMerging();
    void calculateMergeCriteria(Span<float> criterionValues) const;

    void resetRefinementCounters() const;
    void resetEnhancedMergeData(uint32_t currentParticleCount) const;

    Memory<float> _criterionValuesSplit;
    Memory<uint32_t> _particlesIdsToSplit;
    Memory<uint32_t> _particlesSplitCount;
    Memory<float> _criterionValuesMerge;
    Memory<uint32_t> _particlesIdsToMergeFirst;
    Memory<uint32_t> _particlesIdsToMergeSecond;
    Memory<refinement::RefinementData::RemovalState> _removalFlags;
    Memory<uint32_t> _prefixSums;
    Memory<uint32_t> _particlesMergeCount;
    Memory<uint32_t> _particlesIds;
    Memory<uint32_t> _particlesCount;

    // Enhanced merge data with RAII
    Memory<float> _enhancedCriterionValues;
    Memory<uint32_t> _eligibleParticles;
    Memory<uint32_t> _eligibleCount;
    Memory<refinement::MergeState> _states;
    Memory<refinement::MergePair> _pairs;
    Memory<uint32_t> _pairCount;
    Memory<uint32_t> _compactionMap;
    Memory<uint32_t> _newParticleCount;

    refinement::RefinementParameters _refinementParams;
    refinement::RefinementData _refinementData;
    refinement::EnhancedMergeData _enhancedMergeData;
    uint32_t _frameCounter = 0;
    uint32_t _targetParticleCount = 0;
    uint32_t _particlesRemovedInLastMerge = 0;
};

}
