#pragma once

#include <cstdint>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <vector>

#include "../Simulation.cuh"
#include "cuda/Api.cuh"

namespace sph::cuda::physics
{

class SPH_CUDA_API StaticBoundaryDomain
{
public:
    struct Particle
    {
        glm::vec4 position;
        float psi;
    };

    static auto generate(const Simulation::Parameters::Domain& bounds,
                         float particleSpacing,
                         float fluidRestDensity,
                         float fluidSmoothingRadius) -> StaticBoundaryDomain;

    [[nodiscard]] auto getParticles() const -> const std::vector<Particle>&
    {
        return _particles;
    }

    [[nodiscard]] auto getParticleCount() const -> uint32_t
    {
        return static_cast<uint32_t>(_particles.size());
    }

private:
    StaticBoundaryDomain(Simulation::Parameters::Domain bounds, std::vector<Particle> particles);

    static auto generateWallParticles(const Simulation::Parameters::Domain& bounds, float spacing)
        -> std::vector<glm::vec4>;
    static auto computePsi(const std::vector<glm::vec4>& positions, float smoothingRadius, float restDensity)
        -> std::vector<Particle>;
    static auto computeDelta(const std::vector<glm::vec4>& positions, glm::vec4 particlePosition, float smoothingRadius)
        -> float;

    Simulation::Parameters::Domain _bounds;
    std::vector<Particle> _particles {};
};

}
