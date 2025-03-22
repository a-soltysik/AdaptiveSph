#include "RotationHandler.hpp"

#include <glm/ext/vector_float2.hpp>
#include <glm/gtc/constants.hpp>

#include "input_handler/MouseHandler.hpp"

namespace sph
{

RotationHandler::RotationHandler(const Window& window)
    : _window {window}
{
}

auto RotationHandler::getRotation() const -> glm::vec2
{
    const auto delta = _window.getMouseHandler().getCursorDeltaPosition();
    const auto ratio = getPixelsToAngleRatio();

    return {delta.y * ratio.y, delta.x * ratio.x};
}

auto RotationHandler::getPixelsToAngleRatio() const -> glm::vec2
{
    const auto widowSize = _window.getSize();
    return {glm::two_pi<float>() / static_cast<float>(widowSize.x),
            glm::two_pi<float>() / static_cast<float>(widowSize.y)};
}

}
