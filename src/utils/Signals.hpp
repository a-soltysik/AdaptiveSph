#pragma once

#include <panda/Window.h>
#include <panda/utils/Signal.h>

namespace sph::signals
{

struct KeyboardStateChangedData
{
    panda::Window::Id id;
    int key;
    int scancode;
    int action;
    int mods;
};

using KeyboardStateChanged = panda::utils::Signal<KeyboardStateChangedData>;

struct MouseButtonStateChangedData
{
    panda::Window::Id id;
    int button;
    int action;
    int mods;
};

using MouseButtonStateChanged = panda::utils::Signal<MouseButtonStateChangedData>;

struct CursorPositionChangedData
{
    panda::Window::Id id;
    double x;
    double y;
};

using CursorPositionChanged = panda::utils::Signal<CursorPositionChangedData>;

inline auto keyboardStateChanged = KeyboardStateChanged {};
inline auto mouseButtonStateChanged = MouseButtonStateChanged {};
inline auto cursorPositionChanged = CursorPositionChanged {};

}
