#include "MovementHandler.hpp"

#include <GLFW/glfw3.h>

#include "input_handler/KeyboardHandler.hpp"

namespace sph
{

MovementHandler::MovementHandler(const Window& window)
    : _window {window}
{
}

auto MovementHandler::getMovement() const -> Result
{
    auto direction = Result {};
    const auto& keyboardHandler = _window.getKeyboardHandler();

    const auto forwardButton = keyboardHandler.getKeyState(GLFW_KEY_W);
    const auto backButton = keyboardHandler.getKeyState(GLFW_KEY_S);
    const auto rightButton = keyboardHandler.getKeyState(GLFW_KEY_D);
    const auto leftButton = keyboardHandler.getKeyState(GLFW_KEY_A);
    const auto upButton = keyboardHandler.getKeyState(GLFW_KEY_SPACE);
    const auto downButton = keyboardHandler.getKeyState(GLFW_KEY_LEFT_SHIFT);

    using enum KeyboardHandler::State;

    if (forwardButton == Pressed || forwardButton == JustPressed)
    {
        direction.z = direction.z.value_or(0) + 1;
    }
    if (backButton == Pressed || backButton == JustPressed)
    {
        direction.z = direction.z.value_or(0) - 1;
    }
    if (rightButton == Pressed || rightButton == JustPressed)
    {
        direction.x = direction.x.value_or(0) + 1;
    }
    if (leftButton == Pressed || rightButton == JustPressed)
    {
        direction.x = direction.x.value_or(0) - 1;
    }
    if (upButton == Pressed || upButton == JustPressed)
    {
        direction.y = direction.y.value_or(0) + 1;
    }
    if (downButton == Pressed || downButton == JustPressed)
    {
        direction.y = direction.y.value_or(0) - 1;
    }

    return direction;
}

}
