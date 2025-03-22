#pragma once

#include <glm/ext/vector_float2.hpp>

#include "Window.hpp"

namespace sph
{

class RotationHandler
{
public:
    explicit RotationHandler(const Window& window);
    [[nodiscard]] auto getRotation() const -> glm::vec2;

private:
    [[nodiscard]] auto getPixelsToAngleRatio() const -> glm::vec2;

    const Window& _window;
};

}
