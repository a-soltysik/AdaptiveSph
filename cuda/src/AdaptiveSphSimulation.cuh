#pragma once

#include <vector_types.h>

#include <cstdint>
#include <glm/ext/vector_float4.hpp>
#include <vector>

#include "SphSimulation.cuh"
#include "cuda/Simulation.cuh"
#include "cuda/refinement/RefinementParameters.cuh"
#include "refinement/Common.cuh"

namespace sph::cuda
{

class AdaptiveSphSimulation : public SphSimulation
{
public:
    AdaptiveSphSimulation(const Parameters& initialParameters,
                          const std::vector<glm::vec4>& positions,
                          const ParticlesDataBuffer& memory,
                          const refinement::RefinementParameters& refinementParams);

    AdaptiveSphSimulation(const AdaptiveSphSimulation&) = delete;
    AdaptiveSphSimulation(AdaptiveSphSimulation&&) = delete;

    auto operator=(const AdaptiveSphSimulation&) -> AdaptiveSphSimulation& = delete;
    auto operator=(AdaptiveSphSimulation&&) -> AdaptiveSphSimulation& = delete;
    ~AdaptiveSphSimulation() override;

    void update(const Parameters& parameters, float deltaTime) override;
    void identifyAndSplitParticles() const;
    void identifyAndMergeParticles() const;
    void computePrefixSum() const;

private:
    void performAdaptiveRefinement();
    void updateParticleCount();
    [[nodiscard]] auto getBlocksPerGridForParticles(uint32_t count) const -> dim3;

    void resetRefinementCounters() const;

    static auto initializeRefinementData(uint32_t maxParticleCount, float maxBatchSize) -> refinement::RefinementData;

    refinement::RefinementParameters _refinementParams;
    refinement::RefinementData _refinementData;
    uint32_t _frameCounter = 0;
};

}
