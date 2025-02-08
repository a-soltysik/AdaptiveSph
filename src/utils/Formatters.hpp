#pragma once

#include <fmt/base.h>
#include <fmt/format.h>

#include <string_view>

#include "input_handler/KeyboardHandler.hpp"
#include "input_handler/MouseHandler.hpp"

template <>
struct fmt::formatter<sph::KeyboardHandler::State> : formatter<std::string_view>
{
    template <typename FormatContext>
    [[nodiscard]] auto format(sph::KeyboardHandler::State state, FormatContext& ctx) const -> decltype(ctx.out())
    {
        return formatter<std::string_view>::format(getStateName(state), ctx);
    }

    [[nodiscard]] static constexpr auto getStateName(sph::KeyboardHandler::State state) noexcept -> std::string_view
    {
        using namespace std::string_view_literals;
        using enum sph::KeyboardHandler::State;
        switch (state)
        {
        case JustPressed:
            return "JustPressed"sv;
        case Pressed:
            return "Pressed"sv;
        case JustReleased:
            return "JustReleased"sv;
        case Released:
            return "Released"sv;
        default:
            return "UnknownState"sv;
        }
    }
};

template <>
struct fmt::formatter<sph::MouseHandler::ButtonState> : formatter<std::string_view>
{
    template <typename FormatContext>
    [[nodiscard]] auto format(sph::MouseHandler::ButtonState state, FormatContext& ctx) const -> decltype(ctx.out())
    {
        return formatter<std::string_view>::format(getStateName(state), ctx);
    }

    [[nodiscard]] static constexpr auto getStateName(sph::MouseHandler::ButtonState state) noexcept -> std::string_view
    {
        using namespace std::string_view_literals;
        using enum sph::MouseHandler::ButtonState;
        switch (state)
        {
        case JustPressed:
            return "JustPressed"sv;
        case Pressed:
            return "Pressed"sv;
        case JustReleased:
            return "JustReleased"sv;
        case Released:
            return "Released"sv;
        default:
            return "UnknownState"sv;
        }
    }
};
