#pragma once

#include <cstdint>
#include <cuda/Simulation.cuh>
#include <cuda/refinement/RefinementParameters.cuh>
#include <glm/ext/vector_uint3.hpp>
#include <optional>
#include <string>

namespace sph::utils
{
struct InitialParameters
{
    glm::uvec3 particleCount = {40, 40, 25};

    [[nodiscard]] auto getScalarCount() const -> uint32_t
    {
        return particleCount.x * particleCount.y * particleCount.z;
    }
};

struct Configuration
{
    std::optional<InitialParameters> initialParameters;
    std::optional<cuda::Simulation::Parameters> simulationParameters;
    std::optional<cuda::refinement::RefinementParameters> refinementParameters;
};

auto loadConfigurationFromFile(const std::string& filePath) -> std::optional<Configuration>;
auto dumpTemplateConfiguration(const std::string& filePath) -> void;

}
