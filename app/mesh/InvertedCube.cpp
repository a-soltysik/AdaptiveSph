#include "InvertedCube.hpp"

#include <panda/gfx/Context.h>
#include <panda/gfx/Vertex.h>
#include <panda/gfx/object/Mesh.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace sph::mesh::inverted_cube
{
auto create(const panda::gfx::Context& context, std::string name) -> std::unique_ptr<panda::gfx::Mesh>
{
    static constexpr auto vertices = std::array {
        panda::gfx::Vertex {.position {-0.5F, -0.5F, -0.5F}, .normal {0.0F, 0.0F, 1.0F},  .uv {0.0F, 0.0F}},
        panda::gfx::Vertex {.position {0.5F, -0.5F, -0.5F},  .normal {0.0F, 0.0F, 1.0F},  .uv {1.0F, 0.0F}},
        panda::gfx::Vertex {.position {0.5F, 0.5F, -0.5F},   .normal {0.0F, 0.0F, 1.0F},  .uv {1.0F, 1.0F}},
        panda::gfx::Vertex {.position {-0.5F, 0.5F, -0.5F},  .normal {0.0F, 0.0F, 1.0F},  .uv {0.0F, 1.0F}},
        panda::gfx::Vertex {.position {-0.5F, -0.5F, 0.5F},  .normal {0.0F, 0.0F, -1.0F}, .uv {1.0F, 0.0F}},
        panda::gfx::Vertex {.position {-0.5F, 0.5F, 0.5F},   .normal {0.0F, 0.0F, -1.0F}, .uv {1.0F, 1.0F}},
        panda::gfx::Vertex {.position {0.5F, 0.5F, 0.5F},    .normal {0.0F, 0.0F, -1.0F}, .uv {0.0F, 1.0F}},
        panda::gfx::Vertex {.position {0.5F, -0.5F, 0.5F},   .normal {0.0F, 0.0F, -1.0F}, .uv {0.0F, 0.0F}},
        panda::gfx::Vertex {.position {-0.5F, -0.5F, 0.5F},  .normal {-1.0F, 0.0F, 0.0F}, .uv {0.0F, 0.0F}},
        panda::gfx::Vertex {.position {-0.5F, -0.5F, -0.5F}, .normal {-1.0F, 0.0F, 0.0F}, .uv {1.0F, 0.0F}},
        panda::gfx::Vertex {.position {-0.5F, 0.5F, -0.5F},  .normal {-1.0F, 0.0F, 0.0F}, .uv {1.0F, 1.0F}},
        panda::gfx::Vertex {.position {-0.5F, 0.5F, 0.5F},   .normal {-1.0F, 0.0F, 0.0F}, .uv {0.0F, 1.0F}},
        panda::gfx::Vertex {.position {0.5F, -0.5F, 0.5F},   .normal {1.0F, 0.0F, 0.0F},  .uv {1.0F, 0.0F}},
        panda::gfx::Vertex {.position {0.5F, 0.5F, 0.5F},    .normal {1.0F, 0.0F, 0.0F},  .uv {1.0F, 1.0F}},
        panda::gfx::Vertex {.position {0.5F, 0.5F, -0.5F},   .normal {1.0F, 0.0F, 0.0F},  .uv {0.0F, 1.0F}},
        panda::gfx::Vertex {.position {0.5F, -0.5F, -0.5F},  .normal {1.0F, 0.0F, 0.0F},  .uv {0.0F, 0.0F}},
        panda::gfx::Vertex {.position {-0.5F, -0.5F, 0.5F},  .normal {0.0F, 1.0F, 0.0F},  .uv {0.0F, 0.0F}},
        panda::gfx::Vertex {.position {0.5F, -0.5F, 0.5F},   .normal {0.0F, 1.0F, 0.0F},  .uv {1.0F, 0.0F}},
        panda::gfx::Vertex {.position {0.5F, -0.5F, -0.5F},  .normal {0.0F, 1.0F, 0.0F},  .uv {1.0F, 1.0F}},
        panda::gfx::Vertex {.position {-0.5F, -0.5F, -0.5F}, .normal {0.0F, 1.0F, 0.0F},  .uv {0.0F, 1.0F}},
        panda::gfx::Vertex {.position {-0.5F, 0.5F, 0.5F},   .normal {0.0F, -1.0F, 0.0F}, .uv {1.0F, 0.0F}},
        panda::gfx::Vertex {.position {-0.5F, 0.5F, -0.5F},  .normal {0.0F, -1.0F, 0.0F}, .uv {1.0F, 1.0F}},
        panda::gfx::Vertex {.position {0.5F, 0.5F, -0.5F},   .normal {0.0F, -1.0F, 0.0F}, .uv {0.0F, 1.0F}},
        panda::gfx::Vertex {.position {0.5F, 0.5F, 0.5F},    .normal {0.0F, -1.0F, 0.0F}, .uv {0.0F, 0.0F}},
    };
    static constexpr auto indices =
        std::array<uint32_t, 36> {0,  1,  2,  2,  3,  0,  4,  5,  6,  6,  7,  4,  8,  9,  10, 10, 11, 8,
                                  12, 13, 14, 14, 15, 12, 16, 17, 18, 18, 19, 16, 20, 21, 22, 22, 23, 20};

    return std::make_unique<panda::gfx::Mesh>(std::move(name), context.getDevice(), vertices, indices);
}

}  // namespace sph::mesh::inverted_cube