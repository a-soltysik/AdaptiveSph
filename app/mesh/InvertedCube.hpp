#pragma once

#include <panda/gfx/object/Mesh.h>
#include <panda/gfx/object/Object.h>

#include <memory>
#include <string>

namespace sph::mesh::inverted_cube
{
auto create(const panda::gfx::Context& context, std::string name) -> std::unique_ptr<panda::gfx::Mesh>;

};
