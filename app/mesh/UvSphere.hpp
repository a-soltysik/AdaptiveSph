#pragma once

#include <panda/gfx/vulkan/object/Mesh.h>
#include <panda/gfx/vulkan/object/Object.h>

#include <cstdint>
#include <memory>
#include <string>

namespace sph::mesh::uv_sphere
{

struct Shape
{
    float radius;
    uint32_t stacks;
    uint32_t slices;
};

auto create(const panda::gfx::vulkan::Context& context, std::string name, const Shape& shape)
    -> std::unique_ptr<panda::gfx::vulkan::Mesh>;

};
