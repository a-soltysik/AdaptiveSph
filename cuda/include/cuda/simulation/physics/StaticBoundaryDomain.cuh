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

    enum class WallFlags : uint32_t
    {
        None = 0,
        MinX = 1 << 0,
        MaxX = 1 << 1,
        MinY = 1 << 2,
        MaxY = 1 << 3,
        MinZ = 1 << 4,
        MaxZ = 1 << 5,
        All = MinX | MaxX | MinY | MaxY | MinZ | MaxZ
    };

    static auto generate(const Simulation::Parameters::Domain& bounds,
                         float particleSpacing,
                         float fluidRestDensity,
                         float fluidSmoothingRadius,
                         WallFlags wallFlags = WallFlags::All,
                         bool periodicX = false) -> StaticBoundaryDomain;

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

    static auto generateWallParticles(const Simulation::Parameters::Domain& bounds,
                                      float spacing,
                                      WallFlags wallFlags,
                                      bool periodicX)
        -> std::vector<glm::vec4>;
    static auto computePsi(const std::vector<glm::vec4>& positions, float smoothingRadius, float restDensity)
        -> std::vector<Particle>;
    static auto computeDelta(const std::vector<glm::vec4>& positions, glm::vec4 particlePosition, float smoothingRadius)
        -> float;

    Simulation::Parameters::Domain _bounds;
    std::vector<Particle> _particles {};
};

constexpr auto operator|(StaticBoundaryDomain::WallFlags lhs, StaticBoundaryDomain::WallFlags rhs) -> StaticBoundaryDomain::WallFlags
{
    return static_cast<StaticBoundaryDomain::WallFlags>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

constexpr auto operator&(StaticBoundaryDomain::WallFlags lhs, StaticBoundaryDomain::WallFlags rhs) -> StaticBoundaryDomain::WallFlags
{
    return static_cast<StaticBoundaryDomain::WallFlags>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}

constexpr auto operator~(StaticBoundaryDomain::WallFlags flags) -> StaticBoundaryDomain::WallFlags
{
    return static_cast<StaticBoundaryDomain::WallFlags>(~static_cast<uint32_t>(flags));
}

constexpr auto hasFlag(StaticBoundaryDomain::WallFlags flags, StaticBoundaryDomain::WallFlags flag) -> bool
{
    return (flags & flag) == flag;
}

}
