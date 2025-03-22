#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <ios>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace panda::utils
{
template <auto>
using ConstexprChecker = void;

template <class... Ts>
struct overload : Ts...
{
    using Ts::operator()...;
};
template <class... Ts>
overload(Ts...) -> overload<Ts...>;

template <typename T, typename... Rest>
auto hashCombine(size_t& seed, const T& value, const Rest&... rest) -> void
{
    seed ^= std::hash<T> {}(value) + 0x9e3779b9 + (seed << 6U) + (seed >> 2U);
    (hashCombine(seed, rest), ...);
}

template <typename T, size_t N, template <typename, typename...> typename To = std::vector>
auto fromArray(const std::array<T, N>& array) -> To<T>
{
    return {array.cbegin(), array.cend()};
}

template <std::integral T>
constexpr auto maxLengthOfType() -> uint32_t
{
    return std::numeric_limits<T>::digits10 + 2;
}

template <std::integral T>
auto toNumber(std::string_view number) -> std::optional<T>
{
    auto result = T {};
    const auto [ptr, code] {std::from_chars(number.data(), number.data() + number.length(), result)};

    if (code == std::errc::invalid_argument || code == std::errc::result_out_of_range)
    {
        return {};
    }

    return result;
}

template <std::floating_point T>
auto toNumber(std::string_view number) -> std::optional<T>
{
    auto result = T {};
    const auto [ptr, code] {std::from_chars(number.data(), number.data() + number.length(), result)};

    if (code == std::errc::invalid_argument || code == std::errc::result_out_of_range)
    {
        return {};
    }

    return result;
}

template <std::integral T>
auto toString(T number) -> std::string
{
    auto buffer = std::array<char, maxLengthOfType<T>()> {};
    std::to_chars(buffer.data(), buffer.data() + buffer.size(), number);
    auto result = std::string {buffer.cbegin(), buffer.cend()};
    result.erase(std::ranges::find(result, '\0'), result.end());
    return result;
}

auto toString(std::floating_point auto number) -> std::string
{
    auto stream = std::stringstream {};
    stream << number;
    return stream.str();
}

auto toString(std::floating_point auto number, int32_t precision) -> std::string
{
    std::stringstream stream;
    stream << std::fixed << std::setprecision(precision) << number;
    return stream.str();
}

class ScopeGuard
{
public:
    explicit ScopeGuard(std::function<void()> destructor)
        : _destructor {std::move(destructor)}
    {
    }

    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard(ScopeGuard&&) = delete;

    auto operator=(ScopeGuard&& rhs) -> ScopeGuard& = delete;
    auto operator=(const ScopeGuard& rhs) -> ScopeGuard& = delete;

    ~ScopeGuard()
    {
        if (_destructor != nullptr)
        {
            _destructor();
        }
    }

    friend void swap(ScopeGuard& lhs, ScopeGuard& rhs) noexcept
    {
        using std::swap;

        swap(lhs._destructor, rhs._destructor);
    }

private:
    std::function<void()> _destructor;
};
}
