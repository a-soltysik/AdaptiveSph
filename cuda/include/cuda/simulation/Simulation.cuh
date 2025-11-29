#pragma once

#include <cstdint>
#include <glm/common.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <memory>
#include <optional>
#include <vector>

#include "../Api.cuh"
#include "../memory/ImportedMemory.cuh"
#include "../refinement/RefinementParameters.cuh"

namespace sph::cuda
{
namespace physics
{
class StaticBoundaryDomain;
}

struct FluidParticlesData
{
    glm::vec4* positions;
    glm::vec4* velocities;
    glm::vec4* accelerations;
    float* densities;
    float* radii;
    float* smoothingRadii;
    float* masses;

    uint32_t particleCount;
};

struct BoundaryParticlesData
{
    glm::vec4* positions;
    glm::vec4* colors;
    float* psiValues;
    float* viscosityCoefficients;
    float* radii;

    uint32_t particleCount;
};

struct FluidParticlesDataImportedBuffer
{
    const ImportedMemory& positions;
    const ImportedMemory& velocities;
    const ImportedMemory& densities;
    const ImportedMemory& radii;
};

struct BoundaryParticlesDataImportedBuffer
{
    const ImportedMemory& positions;
    const ImportedMemory& radii;
    const ImportedMemory& colors;
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
            float friction = 0.01F;

            [[nodiscard]] auto getTranslation() const noexcept -> glm::vec3
            {
                return (max + min) / 2.F;
            }

            [[nodiscard]] auto getScale() const noexcept -> glm::vec3
            {
                return max - min;
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

            auto operator==(const Domain& rhs) const -> bool = default;
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
    virtual void updateDomain(const Parameters::Domain& domain,
                              const physics::StaticBoundaryDomain& boundaryDomain) = 0;

    [[nodiscard]] virtual auto getFluidParticlesCount() const -> uint32_t = 0;
    [[nodiscard]] virtual auto getBoundaryParticlesCount() const -> uint32_t = 0;

    [[nodiscard]] virtual auto calculateAverageNeighborCount() -> float = 0;

    [[nodiscard]] virtual auto getDensityInfo(float threshold) -> DensityInfo = 0;

    virtual void enableAdaptiveRefinement() = 0;
};

SPH_CUDA_API auto createSimulation(const Simulation::Parameters& parameters,
                                   const std::vector<glm::vec4>& positions,
                                   const FluidParticlesDataImportedBuffer& fluidParticleMemory,
                                   const BoundaryParticlesDataImportedBuffer& boundaryParticleMemory,
                                   const physics::StaticBoundaryDomain& boundaryDomain,
                                   const std::optional<refinement::RefinementParameters>& refinementParams,
                                   uint32_t maxFluidParticleCapacity,
                                   uint32_t maxBoundaryParticleCapacity) -> std::unique_ptr<Simulation>;
}
