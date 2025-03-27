#include "KeyboardHandler.hpp"

#include <GLFW/glfw3.h>
#include <fmt/format.h>
#include <imgui.h>
#include <panda/Logger.h>
#include <panda/utils/Assert.h>
#include <panda/utils/Signals.h>

#include <cstddef>
#include <cstdint>
#include <utility>

#include "Window.hpp"
#include "utils/Formatters.hpp"  // NOLINT(misc-include-cleaner)
#include "utils/Signals.hpp"

namespace
{

void keyboardStateChangedCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    static auto sender = sph::signals::keyboardStateChanged.registerSender();

    const auto windowId = sph::Window::makeId(window);
    panda::log::Debug("Keyboard state for window [{}] changed to {};{};{};{}", windowId, key, scancode, action, mods);

    if (!ImGui::GetIO().WantCaptureKeyboard)
    {
        sender(sph::signals::KeyboardStateChangedData {.id = windowId,
                                                       .key = key,
                                                       .scancode = scancode,
                                                       .action = action,
                                                       .mods = mods});
    }
}

}

namespace sph
{

KeyboardHandler::KeyboardHandler(const Window& window)
    : _window {window}
{
    _states.fill(State::Released);

    [[maybe_unused]] static const auto oldKeyCallback = _window.setKeyCallback(keyboardStateChangedCallback);

    _keyboardStateChangedReceiver = signals::keyboardStateChanged.connect([this](const auto& data) {
        handleKeyboardState(data);
    });

    _newFrameNotifReceiver = panda::utils::signals::gameLoopIterationStarted.connect([this] {
        handleGameLoopIteration();
    });
}

auto KeyboardHandler::getKeyState(int key) const -> State
{
    panda::expect(
        key,
        [this](auto userKey) {
            return userKey < static_cast<int>(_states.size()) && userKey >= 0;
        },
        fmt::format("Key: {} is beyond the size of array", key));

    return _states[static_cast<size_t>(key)];
}

void KeyboardHandler::handleKeyboardState(const signals::KeyboardStateChangedData& data)
{
    if (data.id != _window.getId())
    {
        return;
    }

    if (const auto isCorrectKey = data.key >= 0 && std::cmp_less(data.key, _states.size());
        !panda::shouldBe(isCorrectKey, fmt::format("Key: {} is beyond the size of array", data.key)))
    {
        return;
    }

    if ((_states[static_cast<size_t>(data.key)] == State::JustReleased ||
         _states[static_cast<size_t>(data.key)] == State::Released) &&
        (data.action == GLFW_PRESS || data.action == GLFW_REPEAT))
    {
        _states[static_cast<size_t>(data.key)] = State::JustPressed;
    }
    else if ((_states[static_cast<size_t>(data.key)] == State::JustPressed ||
              _states[static_cast<size_t>(data.key)] == State::Pressed) &&
             data.action == GLFW_RELEASE)
    {
        _states[static_cast<size_t>(data.key)] = State::JustReleased;
    }
}

void KeyboardHandler::handleGameLoopIteration()
{
    for (auto i = uint32_t {}; i < _states.size(); i++)
    {
        if (_states[i] == State::JustReleased)
        {
            _states[i] = State::Released;
        }
        if (_states[i] == State::JustPressed)
        {
            _states[i] = State::Pressed;
        }
    }
}

}
