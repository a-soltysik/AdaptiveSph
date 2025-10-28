#include "panda/gfx/Camera.h"

#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>

namespace panda::gfx
{

auto Camera::setOrthographicProjection(const projection::Orthographic& projection) -> void
{
    _projectionMatrix = glm::mat4 {1.F};
    _projectionMatrix[0][0] = 2.F / (projection.right - projection.left);
    _projectionMatrix[1][1] = 2.F / (projection.bottom - projection.top);
    _projectionMatrix[2][2] = 1.F / (projection.zFar - projection.zNear);
    _projectionMatrix[3][0] = -(projection.right + projection.left) / (projection.right - projection.left);
    _projectionMatrix[3][1] = -(projection.bottom + projection.top) / (projection.bottom - projection.top);
    _projectionMatrix[3][2] = -projection.zNear / (projection.zFar - projection.zNear);
}

auto Camera::setPerspectiveProjection(const projection::Perspective& projection) -> void
{
    const auto tanHalfFovY = glm::tan(projection.fovY / 2.F);
    _projectionMatrix = glm::mat4 {0.F};
    _projectionMatrix[0][0] = 1.F / (projection.aspect * tanHalfFovY);
    _projectionMatrix[1][1] = 1.F / tanHalfFovY;
    _projectionMatrix[2][2] = projection.zFar / (projection.zFar - projection.zNear);
    _projectionMatrix[2][3] = 1.F;
    _projectionMatrix[3][2] = -(projection.zFar * projection.zNear) / (projection.zFar - projection.zNear);
}

auto Camera::getProjection() const noexcept -> const glm::mat4&
{
    return _projectionMatrix;
}

auto Camera::getView() const noexcept -> const glm::mat4&
{
    return _viewMatrix;
}

auto Camera::setViewDirection(const view::Direction& view) -> void
{
    const auto position = glm::vec3 {view.position.x, -view.position.y, view.position.z};
    const auto direction = glm::vec3 {view.direction.x, -view.direction.y, view.direction.z};

    const auto wVec = glm::vec3 {glm::normalize(direction)};
    const auto uVec = glm::vec3 {glm::normalize(glm::cross(wVec, view.up))};
    const auto vVec = glm::vec3 {glm::cross(wVec, uVec)};

    _viewMatrix = glm::mat4 {1.F};
    _viewMatrix[0][0] = uVec.x;
    _viewMatrix[1][0] = uVec.y;
    _viewMatrix[2][0] = uVec.z;
    _viewMatrix[0][1] = vVec.x;
    _viewMatrix[1][1] = vVec.y;
    _viewMatrix[2][1] = vVec.z;
    _viewMatrix[0][2] = wVec.x;
    _viewMatrix[1][2] = wVec.y;
    _viewMatrix[2][2] = wVec.z;
    _viewMatrix[3][0] = -glm::dot(uVec, position);
    _viewMatrix[3][1] = -glm::dot(vVec, position);
    _viewMatrix[3][2] = -glm::dot(wVec, position);

    _inverseViewMatrix = glm::mat4 {1.F};
    _inverseViewMatrix[0][0] = uVec.x;
    _inverseViewMatrix[0][1] = uVec.y;
    _inverseViewMatrix[0][2] = uVec.z;
    _inverseViewMatrix[1][0] = vVec.x;
    _inverseViewMatrix[1][1] = vVec.y;
    _inverseViewMatrix[1][2] = vVec.z;
    _inverseViewMatrix[2][0] = wVec.x;
    _inverseViewMatrix[2][1] = wVec.y;
    _inverseViewMatrix[2][2] = wVec.z;
    _inverseViewMatrix[3][0] = position.x;
    _inverseViewMatrix[3][1] = position.y;
    _inverseViewMatrix[3][2] = position.z;
}

auto Camera::setViewTarget(const view::Target& view) -> void
{
    setViewDirection({.position = view.position, .direction = view.target - view.position, .up = view.up});
}

auto Camera::setViewYXZ(const view::YXZ& view) -> void
{
    const auto position = glm::vec3 {view.position.x, -view.position.y, view.position.z};

    const auto cosZ = glm::cos(view.rotation.z);
    const auto sinZ = glm::sin(view.rotation.z);
    const auto cosX = glm::cos(view.rotation.x);
    const auto sinX = glm::sin(view.rotation.x);
    const auto cosY = glm::cos(view.rotation.y);
    const auto sinY = glm::sin(view.rotation.y);
    const auto uVec =
        glm::vec3 {((cosY * cosZ) + (sinY * sinX * sinZ)), (cosX * sinZ), ((cosY * sinX * sinZ) - (cosZ * sinY))};
    const auto vVec =
        glm::vec3 {((cosZ * sinY * sinX) - (cosY * sinZ)), (cosX * cosZ), ((cosY * cosZ * sinX) + (sinY * sinZ))};
    const auto wVec = glm::vec3 {(cosX * sinY), (-sinX), (cosY * cosX)};
    _viewMatrix = glm::mat4 {1.F};
    _viewMatrix[0][0] = uVec.x;
    _viewMatrix[1][0] = uVec.y;
    _viewMatrix[2][0] = uVec.z;
    _viewMatrix[0][1] = vVec.x;
    _viewMatrix[1][1] = vVec.y;
    _viewMatrix[2][1] = vVec.z;
    _viewMatrix[0][2] = wVec.x;
    _viewMatrix[1][2] = wVec.y;
    _viewMatrix[2][2] = wVec.z;
    _viewMatrix[3][0] = -glm::dot(uVec, position);
    _viewMatrix[3][1] = -glm::dot(vVec, position);
    _viewMatrix[3][2] = -glm::dot(wVec, position);

    _inverseViewMatrix = glm::mat4 {1.F};
    _inverseViewMatrix[0][0] = uVec.x;
    _inverseViewMatrix[0][1] = uVec.y;
    _inverseViewMatrix[0][2] = uVec.z;
    _inverseViewMatrix[1][0] = vVec.x;
    _inverseViewMatrix[1][1] = vVec.y;
    _inverseViewMatrix[1][2] = vVec.z;
    _inverseViewMatrix[2][0] = wVec.x;
    _inverseViewMatrix[2][1] = wVec.y;
    _inverseViewMatrix[2][2] = wVec.z;
    _inverseViewMatrix[3][0] = position.x;
    _inverseViewMatrix[3][1] = position.y;
    _inverseViewMatrix[3][2] = position.z;
}

auto Camera::getInverseView() const noexcept -> const glm::mat4&
{
    return _inverseViewMatrix;
}

}
