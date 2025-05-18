#pragma once

#include <GLFW/glfw3.h>
#include <panda/utils/Signals.h>

#include <array>
#include <cstdint>
#include <glm/ext/vector_double2.hpp>
#include <glm/ext/vector_float2.hpp>

#include "utils/Signals.hpp"

namespace sph
{

class Window;

class MouseHandler
{
public:
    enum class ButtonState : uint8_t
    {
        JustReleased,
        Released,
        Pressed,
        JustPressed
    };

    explicit MouseHandler(const Window& window);

    [[nodiscard]] auto getButtonState(int button) const -> ButtonState;
    [[nodiscard]] auto getCursorPosition() const -> glm::vec2;
    [[nodiscard]] auto getCursorDeltaPosition() const -> glm::vec2;

private:
    void handleMouseButtonState(const signals::MouseButtonStateChangedData& data);
    void handleCursorPosition(const signals::CursorPositionChangedData& data);
    void handleGameLoopIteration();

    std::array<ButtonState, GLFW_MOUSE_BUTTON_LAST> _states {};
    glm::dvec2 _currentPosition {};
    glm::dvec2 _previousPosition {};

    signals::MouseButtonStateChanged::ReceiverT _mouseButtonStateChangedReceiver;
    signals::CursorPositionChanged::ReceiverT _cursorStateChangedReceiver;
    panda::utils::signals::GameLoopIterationStarted::ReceiverT _newFrameNotifReceiver;

    const Window& _window;
};

}
