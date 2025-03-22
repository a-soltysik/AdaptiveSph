#pragma once

// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include "Mesh.h"
#include "Texture.h"

namespace panda::gfx::vulkan
{

class Surface
{
public:
    Surface(const Texture* texture, const Mesh* mesh);

    [[nodiscard]] auto getTexture() const noexcept -> const Texture&;
    [[nodiscard]] auto getMesh() const noexcept -> const Mesh&;

    constexpr auto operator<=>(const Surface&) const noexcept = default;

private:
    const Texture* _texture;
    const Mesh* _mesh;
};

}
