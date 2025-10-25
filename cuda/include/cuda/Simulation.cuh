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
    glm::vec4* accelerations;

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
    const ImportedMemory& accelerations;
};

class SPH_CUDA_API Simulation
{
public:
    struct Parameters
    {
        struct Domain
        {
            glm::vec3 min;
            glm::vec3 max;

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

        enum class TestCase : uint32_t
        {
            None_,
            LidDrivenCavity,
            PoiseuilleFlow,
            TaylorGreenVortex
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
        TestCase testCase = TestCase::None_;

        uint32_t threadsPerBlock;
    };

    virtual ~Simulation() = default;

    virtual void update(float deltaTime) = 0;
    [[nodiscard]] virtual auto getParticlesCount() const -> uint32_t = 0;
    [[nodiscard]] virtual auto calculateAverageNeighborCount() const -> std::pair<float, float> = 0;
    [[nodiscard]] virtual auto updateDensityDeviations() const -> std::vector<float> = 0;
    virtual void setParticleVelocity(uint32_t particleIndex, const glm::vec4& velocity) = 0;
    // NEW: Enhanced data access methods for advanced metrics collection
    [[nodiscard]] virtual auto getParticlePositions() const -> std::vector<glm::vec4> = 0;
    [[nodiscard]] virtual auto getParticleVelocities() const -> std::vector<glm::vec4> = 0;
    [[nodiscard]] virtual auto getParticleDensities() const -> std::vector<float> = 0;
    [[nodiscard]] virtual auto getParticleMasses() const -> std::vector<float> = 0;

    // NEW: Convenience method to get all particle data at once
    struct ParticleDataSnapshot
    {
        std::vector<glm::vec4> positions;
        std::vector<glm::vec4> velocities;
        std::vector<float> densities;
        std::vector<float> masses;
        uint32_t particleCount;
    };

    [[nodiscard]] virtual auto getParticleDataSnapshot() const -> ParticleDataSnapshot = 0;

    // NEW: Performance timing interface
    [[nodiscard]] virtual auto getLastCudaComputationTime() const -> float = 0;
};

SPH_CUDA_API auto createSimulation(const Simulation::Parameters& parameters,
                                   const std::vector<glm::vec4>& positions,
                                   const ParticlesDataBuffer& memory,
                                   const refinement::RefinementParameters& refinementParams)
    -> std::unique_ptr<Simulation>;

}
