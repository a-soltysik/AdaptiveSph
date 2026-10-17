#pragma once

#include <panda/utils/Assert.h>

#include <../../cuda/include/cuda/simulation/Simulation.cuh>
#include <cstdint>
#include <cuda/refinement/RefinementParameters.cuh>
#include <glm/ext/vector_uint3.hpp>
#include <limits>
#include <optional>
#include <string>

namespace sph::utils
{
struct InitialParameters
{
    glm::uvec3 particleCount = {40, 40, 25};

    [[nodiscard]] auto getScalarCount() const -> uint32_t
    {
        constexpr auto maxCount = std::numeric_limits<uint32_t>::max();
        panda::expect(particleCount.x > 0 && particleCount.y > 0 && particleCount.z > 0,
                      "initialParameters.particleCount axes must be greater than zero");
        panda::expect(particleCount.x <= maxCount / particleCount.y &&
                          particleCount.x * particleCount.y <= maxCount / particleCount.z,
                      "initialParameters.particleCount product exceeds the supported range");
        return particleCount.x * particleCount.y * particleCount.z;
    }
};

inline constexpr float initialParticleMargin = 0.15F;

struct RenderingParameters
{
    bool renderBoundaryParticles = false;
};

struct Configuration
{
    std::optional<InitialParameters> initialParameters;
    std::optional<cuda::Simulation::Parameters> simulationParameters;
    std::optional<cuda::refinement::RefinementParameters> refinementParameters;
    std::optional<RenderingParameters> renderingParameters;
};

auto loadConfigurationFromFile(const std::string& filePath) -> std::optional<Configuration>;
auto validateConfiguration(const Configuration& configuration) -> void;
auto dumpTemplateConfiguration(const std::string& filePath) -> void;

}
