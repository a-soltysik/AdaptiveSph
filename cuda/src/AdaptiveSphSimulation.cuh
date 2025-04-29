#pragma once

#include "SphSimulation.cuh"
#include "cuda/refinement/RefinementParameters.cuh"
#include "refinement/Common.cuh"

namespace sph::cuda
{

/**
 * Extended SPH simulation that adds adaptive particle refinement capabilities
 * allowing dynamic resolution adjustment based on physical criteria
 */
class AdaptiveSphSimulation : public SphSimulation
{
public:
    AdaptiveSphSimulation(const Parameters& initialParameters,
                          const std::vector<glm::vec4>& positions,
                          const ParticlesDataBuffer& memory,
                          const refinement::RefinementParameters& refinementParams);

    ~AdaptiveSphSimulation() override;

    void update(const Parameters& parameters, float deltaTime) override;
    void identifyAndSplitParticles() const;
    void identifyAndMergeParticles() const;
    void computePrefixSum() const;

private:
    /**
     * Perform adaptive refinement based on configured criteria
     */
    void performAdaptiveRefinement(float deltaTime);

    /**
     * Update particle count from device to host
     */
    void updateParticleCount();

    /**
     * Get block count for a specific number of particles
     */
    [[nodiscard]] auto getBlocksPerGridForParticles(uint32_t count) const -> dim3;

    /**
     * Reset counters for refinement operations
     */
    void resetRefinementCounters() const;

    static refinement::RefinementData initializeRefinementData(uint32_t maxParticleCount, float maxBatchSize);

    refinement::RefinementParameters _refinementParams;
    refinement::RefinementData _refinementData;
    uint32_t _frameCounter = 0;
};

}  // namespace sph::cuda
