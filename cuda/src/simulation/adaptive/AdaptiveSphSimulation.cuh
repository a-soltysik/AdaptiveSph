#pragma once

#include <vector_types.h>

#include <cstdint>
#include <glm/ext/vector_float4.hpp>
#include <vector>

#include "cuda/Simulation.cuh"
#include "cuda/refinement/RefinementParameters.cuh"
#include "refinement/Common.cuh"
#include "simulation/SphSimulation.cuh"
#include "utils/DeviceValue.cuh"

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
    void calculateMergeCriteria(float* criterionValues) const;
    void resetRefinementCounters();
    void identifyAndSplitParticles(uint32_t removedParticles);
    void identifyAndMergeParticles();

    refinement::RefinementParameters _refinementParams;
    refinement::RefinementData _refinementData;
    uint32_t _frameCounter = 0;
    uint32_t _targetParticleCount = 0;
};

}
