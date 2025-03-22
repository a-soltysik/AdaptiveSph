#pragma once

#include <fmt/base.h>
#include <fmt/format.h>

#include <concepts>
#include <cstdlib>
#include <optional>
#include <source_location>
#include <string_view>
#include <type_traits>
#include <utility>

#include "panda/Logger.h"

namespace panda
{

template <typename T, typename = decltype(std::declval<T>().value), typename = decltype(std::declval<T>().result)>
struct ResultHelper
{
    using Ok = decltype(std::declval<T>().value);
    using Error = decltype(std::declval<T>().result);
};

template <typename T>
concept Result = requires {
    typename ResultHelper<T>::Ok;
    typename ResultHelper<T>::Error;
};

[[noreturn]] inline auto panic() noexcept
{
    std::abort();
}

[[noreturn]] inline auto panic(std::string_view message,
                               std::source_location location = std::source_location::current()) noexcept
{
    log::internal::LogDispatcher::log(log::Level::Error, fmt::format("Terminal error: {}", message), location);
    panic();
}

template <typename T>
auto expect(T&& result,
            const std::equality_comparable_with<T> auto& expected,
            std::string_view message,
            std::source_location location = std::source_location::current()) noexcept -> T
{
    if (result != expected) [[unlikely]]
    {
        if constexpr (fmt::is_formattable<std::decay_t<T>>() || std::is_enum<std::decay_t<T>>())
        {
            panic(fmt::format("{}: {}", message, result), location);
        }
        else
        {
            panic(message, location);
        }
    }
    return std::forward<T>(result);
}

template <Result T>
auto expect(T&& result,
            const typename ResultHelper<T>::Error& expected,
            std::string_view message,
            std::source_location location = std::source_location::current()) noexcept -> typename ResultHelper<T>::Ok
{
    if (result.result != expected) [[unlikely]]
    {
        panic(fmt::format("{}: {}", message, result.result), location);
    }
    return std::forward<typename ResultHelper<T>::Ok>(result.value);
}

template <typename T>
auto expect(T&& value,
            std::invocable<T> auto predicate,
            std::string_view message,
            std::source_location location = std::source_location::current()) noexcept -> T
{
    if (!predicate(std::forward<T>(value))) [[unlikely]]
    {
        panic(message, location);
    }
    return value;
}

auto expect(std::convertible_to<bool> auto&& result,
            std::string_view message,
            std::source_location location = std::source_location::current()) noexcept
{
    if (!result) [[unlikely]]
    {
        panic(message, location);
    }

    return result;
}

auto expectNot(std::convertible_to<bool> auto&& result,
               std::string_view message,
               std::source_location location = std::source_location::current()) noexcept
{
    if (result) [[unlikely]]
    {
        panic(message, location);
    }
    return result;
}

template <typename T>
auto expectNot(T&& result,
               const std::equality_comparable_with<T> auto& expected,
               std::string_view message,
               std::source_location location = std::source_location::current()) noexcept -> T
{
    if (result == expected) [[unlikely]]
    {
        if constexpr (fmt::is_formattable<std::decay_t<T>>() || std::is_enum<std::decay_t<T>>())
        {
            panic(fmt::format("{}: {}", message, result), location);
        }
        else
        {
            panic(message, location);
        }
    }
    return std::forward<T>(result);
}

template <Result T>
auto expectNot(T&& result,
               const typename ResultHelper<T>::Error& expected,
               std::string_view message,
               std::source_location location = std::source_location::current()) noexcept -> typename ResultHelper<T>::Ok
{
    if (result.result == expected) [[unlikely]]
    {
        panic(fmt::format("{}: {}", message, result.result), location);
    }
    return std::forward<typename ResultHelper<T>::Ok>(result.value);
}

template <typename T>
auto expectNot(T&& value,
               std::invocable<T> auto predicate,
               std::string_view message,
               std::source_location location = std::source_location::current()) noexcept -> T
{
    if (predicate(std::forward<T>(value))) [[unlikely]]
    {
        panic(message, location);
    }
    return value;
}

template <typename T>
auto expect(std::optional<T>&& value,
            std::string_view message,
            std::source_location location = std::source_location::current()) noexcept -> T
{
    if (!value.has_value()) [[unlikely]]
    {
        panic(message, location);
    }
    return std::move(value).value();
}

template <typename T>
auto shouldBe(T&& result,
              const std::equality_comparable_with<T> auto& expected,
              std::string_view message,
              std::source_location location = std::source_location::current()) noexcept -> bool
{
    if (result != expected) [[unlikely]]
    {
        if constexpr (fmt::is_formattable<std::decay_t<T>>() || std::is_enum<std::decay_t<T>>())
        {
            log::internal::LogDispatcher::log(log::Level::Error,
                                              fmt::format("{}: {}", message, std::forward<T>(result)),
                                              location);
        }
        else
        {
            log::internal::LogDispatcher::log(log::Level::Error, fmt::format("{}", message), location);
        }
        return false;
    }
    return true;
}

auto shouldBe(std::convertible_to<bool> auto&& result,
              std::string_view message,
              std::source_location location = std::source_location::current()) noexcept -> bool
{
    if (!result) [[unlikely]]
    {
        log::internal::LogDispatcher::log(log::Level::Error, fmt::format("{}", message), location);
        return false;
    }
    return true;
}

template <typename T>
[[nodiscard]] auto shouldBe(T&& value,
                            std::invocable<T> auto predicate,
                            std::string_view message,
                            std::source_location location = std::source_location::current()) noexcept -> bool
{
    if (!predicate(std::forward<T>(value))) [[unlikely]]
    {
        log::internal::LogDispatcher::log(log::Level::Error, fmt::format("{}", message), location);
        return false;
    }
    return true;
}

template <typename T>
auto shouldNotBe(T&& result,
                 const std::equality_comparable_with<T> auto& expected,
                 std::string_view message,
                 std::source_location location = std::source_location::current()) noexcept -> bool
{
    if (result == expected) [[unlikely]]
    {
        if constexpr (fmt::is_formattable<std::decay_t<T>>() || std::is_enum<std::decay_t<T>>())
        {
            log::internal::LogDispatcher::log(log::Level::Error,
                                              fmt::format("{}: {}", message, std::forward<T>(result)),
                                              location);
        }
        else
        {
            log::internal::LogDispatcher::log(log::Level::Error, fmt::format("{}", message), location);
        }
        return false;
    }
    return true;
}

}

#if NDEBUG
#    define DEBUG_EXPECT(condition) ((void) 0);
#    define EXPECT(condition) panda::shouldBe(condition, #condition);
#else
#    define DEBUG_EXPECT(condition) panda::expect(condition, #condition);
#    define EXPECT(condition) DEBUG_EXPECT(condition);
#endif
