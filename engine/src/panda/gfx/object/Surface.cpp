#include "panda/gfx/object/Surface.h"

#include "panda/gfx/object/Mesh.h"
#include "panda/gfx/object/Texture.h"

namespace panda::gfx
{

Surface::Surface(const Texture* texture, const Mesh* mesh)
    : _texture {texture},
      _mesh {mesh}
{
}

auto Surface::getTexture() const noexcept -> const Texture&
{
    return *_texture;
}

auto Surface::getMesh() const noexcept -> const Mesh&
{
    return *_mesh;
}

}
