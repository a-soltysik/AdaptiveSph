#pragma once

#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/vector_float3.hpp>

namespace panda::gfx
{

namespace projection
{

struct Orthographic
{
    float left;
    float right;
    float top;
    float bottom;
    float zNear;
    float zFar;
};

struct Perspective
{
    float fovY;
    float aspect;
    float zNear;
    float zFar;
};

}

namespace view
{

struct Direction
{
    glm::vec3 position;
    glm::vec3 direction;
    glm::vec3 up = glm::vec3 {0.F, -1.F, 0.F};
};

struct Target
{
    glm::vec3 position;
    glm::vec3 target;
    glm::vec3 up = glm::vec3 {0.F, -1.F, 0.F};
};

struct YXZ
{
    glm::vec3 position;
    glm::vec3 rotation;
};

}

class Camera
{
public:
    auto setOrthographicProjection(const projection::Orthographic& projection) -> void;
    auto setPerspectiveProjection(const projection::Perspective& projection) -> void;
    auto setViewDirection(const view::Direction& view) -> void;
    auto setViewTarget(const view::Target& view) -> void;
    auto setViewYXZ(const view::YXZ& view) -> void;

    [[nodiscard]] auto getProjection() const noexcept -> const glm::mat4&;
    [[nodiscard]] auto getView() const noexcept -> const glm::mat4&;
    [[nodiscard]] auto getInverseView() const noexcept -> const glm::mat4&;

private:
    glm::mat4 _projectionMatrix {1.F};
    glm::mat4 _viewMatrix {1.F};
    glm::mat4 _inverseViewMatrix {1.F};
};

}
