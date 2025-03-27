#include "MovementHandler.hpp"

#include <GLFW/glfw3.h>

#include <optional>

#include "input_handler/KeyboardHandler.hpp"

namespace sph
{

MovementHandler::MovementHandler(const Window& window)
    : _window {window}
{
}

auto MovementHandler::getMovement() const -> Result
{
    return Result {.x = getXMovement(), .y = getYMovement(), .z = getZMovement()};
}

auto MovementHandler::getXMovement() const -> std::optional<float>
{
    const auto& keyboardHandler = _window.getKeyboardHandler();
    const auto rightButton = keyboardHandler.getKeyState(GLFW_KEY_D);
    const auto leftButton = keyboardHandler.getKeyState(GLFW_KEY_A);

    auto direction = std::optional<float> {};
    using enum KeyboardHandler::State;

    if (rightButton == Pressed || rightButton == JustPressed)
    {
        direction = direction.value_or(0) + 1;
    }
    if (leftButton == Pressed || rightButton == JustPressed)
    {
        direction = direction.value_or(0) - 1;
    }

    return direction;
}

auto MovementHandler::getYMovement() const -> std::optional<float>
{
    const auto& keyboardHandler = _window.getKeyboardHandler();
    const auto upButton = keyboardHandler.getKeyState(GLFW_KEY_SPACE);
    const auto downButton = keyboardHandler.getKeyState(GLFW_KEY_LEFT_SHIFT);

    auto direction = std::optional<float> {};
    using enum KeyboardHandler::State;
    if (upButton == Pressed || upButton == JustPressed)
    {
        direction = direction.value_or(0) + 1;
    }
    if (downButton == Pressed || downButton == JustPressed)
    {
        direction = direction.value_or(0) - 1;
    }

    return direction;
}

auto MovementHandler::getZMovement() const -> std::optional<float>
{
    const auto& keyboardHandler = _window.getKeyboardHandler();

    const auto forwardButton = keyboardHandler.getKeyState(GLFW_KEY_W);
    const auto backButton = keyboardHandler.getKeyState(GLFW_KEY_S);

    auto direction = std::optional<float> {};
    using enum KeyboardHandler::State;
    if (forwardButton == Pressed || forwardButton == JustPressed)
    {
        direction = direction.value_or(0) + 1;
    }
    if (backButton == Pressed || backButton == JustPressed)
    {
        direction = direction.value_or(0) - 1;
    }

    return direction;
}

}
