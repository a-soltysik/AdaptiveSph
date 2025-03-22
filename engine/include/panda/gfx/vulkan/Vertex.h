#pragma once

// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include <array>
#include <cstddef>
#include <glm/ext/vector_float2.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/gtx/hash.hpp>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_structs.hpp>

namespace panda::gfx::vulkan
{

struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;

    constexpr auto operator==(const Vertex& rhs) const noexcept -> bool = default;

    static constexpr auto getBindingDescription() -> vk::VertexInputBindingDescription
    {
        return vk::VertexInputBindingDescription {0, sizeof(Vertex), vk::VertexInputRate::eVertex};
    }

    static constexpr auto getAttributeDescriptions() -> std::array<vk::VertexInputAttributeDescription, 3>
    {
        return {
            vk::VertexInputAttributeDescription {0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, position)},
            vk::VertexInputAttributeDescription {1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal)  },
            vk::VertexInputAttributeDescription {2, 0, vk::Format::eR32G32Sfloat,    offsetof(Vertex, uv)      }
        };
    }
};

}
