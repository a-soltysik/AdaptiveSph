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

namespace panda::gfx
{

struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;

    constexpr auto operator==(const Vertex& rhs) const noexcept -> bool = default;

    static constexpr auto getBindingDescription() -> vk::VertexInputBindingDescription
    {
        return vk::VertexInputBindingDescription {.binding = 0,
                                                  .stride = sizeof(Vertex),
                                                  .inputRate = vk::VertexInputRate::eVertex};
    }

    static constexpr auto getAttributeDescriptions() -> std::array<vk::VertexInputAttributeDescription, 3>
    {
        return {
            vk::VertexInputAttributeDescription {.location = 0,
                                                 .binding = 0,
                                                 .format = vk::Format::eR32G32B32Sfloat,
                                                 .offset = offsetof(Vertex, position)},
            vk::VertexInputAttributeDescription {.location = 1,
                                                 .binding = 0,
                                                 .format = vk::Format::eR32G32B32Sfloat,
                                                 .offset = offsetof(Vertex, normal)  },
            vk::VertexInputAttributeDescription {.location = 2,
                                                 .binding = 0,
                                                 .format = vk::Format::eR32G32Sfloat,
                                                 .offset = offsetof(Vertex, uv)      }
        };
    }
};

}
