#pragma once

#include <cstdint>
#include <glm/common.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <memory>
#include <vector>

#include "Api.cuh"
#include "ImportedMemory.cuh"
#include "refinement/RefinementParameters.cuh"

namespace sph::cuda
{
struct ParticlesData
{
    glm::vec4* positions;
    glm::vec4* predictedPositions;
    glm::vec4* velocities;
    float* densities;
    float* nearDensities;
    float* pressures;
    float* radiuses;
    float* smoothingRadiuses;
    float* masses;
    float* densityDeviations;

    uint32_t particleCount;
};

struct ParticlesDataBuffer
{
    const ImportedMemory& positions;
    const ImportedMemory& predictedPositions;
    const ImportedMemory& velocities;
    const ImportedMemory& densities;
    const ImportedMemory& nearDensities;
    const ImportedMemory& pressures;
    const ImportedMemory& radiuses;
    const ImportedMemory& smoothingRadiuses;
    const ImportedMemory& masses;
    const ImportedMemory& densityDeviations;
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

            auto fromTransform(glm::vec3 translation, glm::vec3 scale) -> Domain
            {
                return {.min = translation - scale / 2.F, .max = translation + scale / 2.F};
            }
        };

        Domain domain;
        glm::vec3 gravity;
        float restDensity;
        float pressureConstant;
        float nearPressureConstant;
        float restitution;
        float viscosityConstant;
        float maxVelocity;
        float baseSmoothingRadius;
        float baseParticleRadius;
        float baseParticleMass;
        float lidVelocity = 0.F;

        uint32_t threadsPerBlock = 256;
    };

    virtual ~Simulation() = default;

    virtual void update(float deltaTime) = 0;

    [[nodiscard]] virtual auto getParticlesCount() const -> uint32_t = 0;

    [[nodiscard]] virtual auto calculateAverageNeighborCount() const -> float = 0;

    [[nodiscard]] virtual auto updateDensityDeviations() const -> std::vector<float> = 0;

    virtual void setParticleVelocity(uint32_t particleIndex, const glm::vec4& velocity) = 0;
};

SPH_CUDA_API auto createSimulation(const Simulation::Parameters& parameters,
                                   const std::vector<glm::vec4>& positions,
                                   const ParticlesDataBuffer& memory,
                                   const refinement::RefinementParameters& refinementParams)
    -> std::unique_ptr<Simulation>;
}
