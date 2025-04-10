#pragma once

#include <cstdint>
#include <glm/common.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <memory>
#include <vector>

#include "Api.cuh"
#include "ImportedMemory.cuh"

namespace sph::cuda
{

struct ParticlesData
{
    glm::vec4* positions;
    glm::vec4* predictedPositions;
    glm::vec4* velocities;
    glm::vec4* forces;
    float* densities;
    float* nearDensities;
    float* pressures;
    float* radiuses;

    uint32_t particleCount;
};

struct ParticlesDataBuffer
{
    const ImportedMemory& positions;
    const ImportedMemory& predictedPositions;
    const ImportedMemory& velocities;
    const ImportedMemory& forces;
    const ImportedMemory& densities;
    const ImportedMemory& nearDensities;
    const ImportedMemory& pressures;
    const ImportedMemory& radiuses;
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
        float smoothingRadius;
        float viscosityConstant;
        float surfaceTensionCoefficient;
        float maxVelocity;
        float particleRadius;

        uint32_t threadsPerBlock;
    };

    virtual ~Simulation() = default;

    virtual void update(const Parameters& parameters, float deltaTime) = 0;
};

SPH_CUDA_API auto createSimulation(const Simulation::Parameters& data,
                                   const std::vector<glm::vec4>& positions,
                                   const ParticlesDataBuffer& memory) -> std::unique_ptr<Simulation>;

}
