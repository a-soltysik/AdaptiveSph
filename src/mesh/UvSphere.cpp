#include "UvSphere.hpp"

#include <panda/gfx/vulkan/Context.h>
#include <panda/gfx/vulkan/Vertex.h>
#include <panda/gfx/vulkan/object/Mesh.h>

#include <cstddef>
#include <cstdint>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/trigonometric.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sph::mesh::uv_sphere
{
namespace
{

auto createVertices(const Shape& shape) -> std::vector<panda::gfx::vulkan::Vertex>
{
    auto vertices = std::vector<panda::gfx::vulkan::Vertex> {};
    vertices.reserve(static_cast<size_t>(shape.stacks + 1) * (shape.slices + 1));
    for (auto i = uint32_t {}; i <= shape.stacks; ++i)
    {
        const auto phi = glm::pi<float>() * static_cast<float>(i) / static_cast<float>(shape.stacks);
        for (auto j = uint32_t {}; j <= shape.slices; ++j)
        {
            const auto theta = glm::two_pi<float>() * static_cast<float>(j) / static_cast<float>(shape.slices);

            const auto x = shape.radius * glm::sin(phi) * glm::cos(theta);
            const auto y = shape.radius * glm::cos(phi);
            const auto z = shape.radius * glm::sin(phi) * glm::sin(theta);

            const auto u = static_cast<float>(j) / static_cast<float>(shape.slices);
            const auto v = static_cast<float>(i) / static_cast<float>(shape.stacks);

            vertices.push_back({
                .position {x, y, z},
                .normal = {x / shape.radius, y / shape.radius, z / shape.radius},
                .uv {u, v}
            });
        }
    }
    return vertices;
}

auto createIndices(const Shape& shape) -> std::vector<uint32_t>
{
    auto indices = std::vector<uint32_t> {};
    indices.reserve(static_cast<size_t>(shape.stacks * shape.slices) * 3);
    for (auto i = uint32_t {}; i < shape.stacks; ++i)
    {
        for (auto j = uint32_t {}; j < shape.slices; ++j)
        {
            const auto first = (i * (shape.slices + 1)) + j;
            const auto second = first + shape.slices + 1;

            indices.push_back(first);
            indices.push_back(first + 1);
            indices.push_back(second);

            indices.push_back(second);
            indices.push_back(first + 1);
            indices.push_back(second + 1);
        }
    }
    return indices;
}

}

auto create(const panda::gfx::vulkan::Context& context, std::string name, const Shape& shape)
    -> std::unique_ptr<panda::gfx::vulkan::Mesh>
{
    return std::make_unique<panda::gfx::vulkan::Mesh>(std::move(name),
                                                      context.getDevice(),
                                                      createVertices(shape),
                                                      createIndices(shape));
}
}
