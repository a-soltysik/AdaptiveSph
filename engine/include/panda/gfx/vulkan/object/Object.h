#pragma once

// clang-format off
#include "panda/utils/Assert.h" // NOLINT(misc-include-cleaner)
// clang-format on

#include <cstddef>
#include <glm/ext/vector_float3.hpp>
#include <string>
#include <vector>

#include "Surface.h"
#include "panda/Common.h"

namespace panda::gfx::vulkan
{

struct Transform
{
    glm::vec3 translation {};
    glm::vec3 scale {1.F, 1.F, 1.F};
    glm::vec3 rotation {};
};

class Scene;
class Context;

class Object
{
public:
    using Id = size_t;

    static auto getNextId() -> Id;

    explicit Object(std::string name);
    PD_DELETE_ALL(Object);
    ~Object() noexcept = default;

    [[nodiscard]] auto getId() const noexcept -> Id;
    [[nodiscard]] auto getName() const noexcept -> const std::string&;
    auto addSurface(const Surface& surface) -> void;
    [[nodiscard]] auto getSurfaces() const noexcept -> const std::vector<Surface>&;

    Transform transform;

private:
    inline static Id currentId = 0;
    std::vector<Surface> surfaces;
    std::string _name;
    Id _id;
};

}
