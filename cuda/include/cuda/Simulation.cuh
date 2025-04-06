#pragma once

#include <glm/common.hpp>
#include <glm/ext/vector_float3.hpp>
#include <memory>
#include <vector>

#include "Api.cuh"
#include "ImportedMemory.cuh"

namespace sph::cuda
{

struct ParticleData
{
    alignas(16) glm::vec3 position;
    alignas(16) glm::vec3 predictedPosition;
    alignas(16) glm::vec3 velocity;
    alignas(16) glm::vec3 force;
    alignas(4) float mass;
    alignas(4) float density;
    alignas(4) float nearDensity;
    alignas(4) float pressure;
    alignas(4) float radius;
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
        float particleMass;

        uint32_t threadsPerBlock;
    };

    virtual ~Simulation() = default;

    virtual void update(const Parameters& parameters, float deltaTime) = 0;
};

SPH_CUDA_API auto createSimulation(const Simulation::Parameters& data,
                                   const std::vector<glm::vec3>& positions,
                                   const ImportedMemory& memory) -> std::unique_ptr<Simulation>;

}
