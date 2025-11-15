#pragma once

#include <cstdint>
#include <glm/common.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <memory>
#include <optional>
#include <vector>

#include "Api.cuh"
#include "ImportedMemory.cuh"
#include "refinement/RefinementParameters.cuh"

namespace sph::cuda
{
struct ParticlesData
{
    glm::vec4* positions;
    glm::vec4* velocities;
    glm::vec4* accelerations;
    float* densities;
    float* radiuses;
    float* smoothingRadiuses;
    float* masses;

    uint32_t particleCount;
};

struct ParticlesDataBuffer
{
    const ImportedMemory& positions;
    const ImportedMemory& velocities;
    const ImportedMemory& accelerations;
    const ImportedMemory& densities;
    const ImportedMemory& radiuses;
    const ImportedMemory& smoothingRadiuses;
    const ImportedMemory& masses;
};

class SPH_CUDA_API Simulation
{
public:
    struct Parameters
    {
        struct Domain
        {
            glm::vec3 min = {-1.F, -1.F, -1.F};
            glm::vec3 max = {1.F, 1.F, 1.F};

            [[nodiscard]] auto getTranslation() const noexcept -> glm::vec3
            {
                return (max + min) / 2.F;
            }

            [[nodiscard]] auto getScale() const noexcept -> glm::vec3
            {
                return glm::abs(max - min);
            }

            [[nodiscard]] auto getVolume() const noexcept -> float
            {
                const auto scale = getScale();
                return scale.x * scale.y * scale.z;
            }

            static auto fromTransform(glm::vec3 translation, glm::vec3 scale) -> Domain
            {
                return {.min = translation - scale / 2.F, .max = translation + scale / 2.F};
            }
        };

        Domain domain;
        glm::vec3 gravity;
        float speedOfSound;
        float restDensity;
        float restitution;
        float viscosityConstant;
        float surfaceTensionConstant;
        float maxVelocity;
        float baseSmoothingRadius;
        float baseParticleRadius;
        float baseParticleMass;

        uint32_t threadsPerBlock = 256;
    };

    struct DensityInfo
    {
        float restDensity;
        float minDensity;
        float maxDensity;
        float averageDensity;
        uint32_t underDensityCount;
        uint32_t normalDensityCount;
        uint32_t overDensityCount;
    };

    virtual ~Simulation() = default;

    virtual void update(float deltaTime) = 0;

    [[nodiscard]] virtual auto getParticlesCount() const -> uint32_t = 0;

    [[nodiscard]] virtual auto calculateAverageNeighborCount() const -> float = 0;

    [[nodiscard]] virtual auto getDensityInfo(float threshold) const -> DensityInfo = 0;
};

SPH_CUDA_API auto createSimulation(const Simulation::Parameters& parameters,
                                   const std::vector<glm::vec4>& positions,
                                   const ParticlesDataBuffer& memory,
                                   const std::optional<refinement::RefinementParameters>& refinementParams,
                                   uint32_t initialParticleCount) -> std::unique_ptr<Simulation>;
}
