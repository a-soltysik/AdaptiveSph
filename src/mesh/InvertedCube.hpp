#pragma once

#include <panda/gfx/vulkan/object/Mesh.h>
#include <panda/gfx/vulkan/object/Object.h>

#include <memory>
#include <string>

namespace sph::mesh::inverted_cube
{
auto create(const panda::gfx::vulkan::Context& context, std::string name) -> std::unique_ptr<panda::gfx::vulkan::Mesh>;

};
