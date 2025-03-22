#pragma once

// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <glm/detail/qualifier.hpp>
#include <glm/ext/vector_float2.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "panda/utils/Concepts.h"
#include "panda/utils/Utils.h"

namespace panda::gfx::vulkan
{

namespace internal
{

template <typename T>
concept UboStruct = requires {
    { T {}.alignment() } -> std::convertible_to<size_t>;
    typename utils::ConstexprChecker<T {}.alignment()>;
};

template <typename T, size_t N>
using ArrayOf = T[N];

template <typename T>
concept ScalarOrVec = utils::Scalar<T> || utils::Vec<T>;

}

template <typename T>
requires(!std::is_default_constructible_v<T> || !std::is_standard_layout_v<T>)
consteval auto alignment(T = {}) -> size_t;

template <internal::UboStruct T>
consteval auto alignment(T = {}) noexcept -> size_t
{
    return T {}.alignment();
};

template <utils::Scalar T>
[[nodiscard]] consteval auto alignment([[maybe_unused]] T = {}) noexcept -> size_t
{
    return sizeof(T);
}

static_assert(alignment<uint32_t>() == 4);

template <template <glm::length_t, typename, glm::qualifier> typename V, glm::length_t L, typename T, glm::qualifier Q>
requires utils::Vec<V<L, T, Q>>
[[nodiscard]] consteval auto alignment([[maybe_unused]] V<L, T, Q> = {}) noexcept -> size_t
{
    if constexpr (L == 2)
    {
        return 2 * sizeof(T);
    }
    return 4 * sizeof(T);
}

static_assert(alignment(glm::vec2 {}) == 8);
static_assert(alignment(glm::vec3 {}) == 16);
static_assert(alignment(glm::vec4 {}) == 16);

template <internal::ScalarOrVec T, size_t N>
[[nodiscard]] consteval auto alignment([[maybe_unused]] const internal::ArrayOf<T, N> = {}) noexcept -> size_t
{
    return std::max(sizeof(T), alignment(glm::vec4 {}));
}

static_assert(alignment<uint32_t, 5>({}) == 16);

template <internal::UboStruct T, size_t N>
[[nodiscard]] consteval auto alignment([[maybe_unused]] const internal::ArrayOf<T, N> = {}) noexcept -> size_t
{
    return T {}.alignment();
}

template <internal::ScalarOrVec T, size_t N>
[[nodiscard]] consteval auto alignment([[maybe_unused]] const std::array<T, N> = {}) noexcept -> size_t
{
    return std::max(sizeof(T), alignment(glm::vec4 {}));
}

static_assert(alignment(std::array<uint32_t, 5> {}) == 16);

template <internal::UboStruct T, size_t N>
[[nodiscard]] consteval auto alignment([[maybe_unused]] const std::array<T, N> = {}) noexcept -> size_t
{
    return T {}.alignment();
}

template <template <glm::length_t, glm::length_t, typename, glm::qualifier> typename M,
          glm::length_t C,
          glm::length_t R,
          typename T,
          glm::qualifier Q>
requires utils::Mat<M<C, R, T, Q>>
[[nodiscard]] consteval auto alignment([[maybe_unused]] M<C, R, T, Q> = {}) noexcept -> size_t
{
    return alignment(std::array<glm::vec<R, T, Q>, static_cast<size_t>(C)> {});
}

static_assert(alignment(glm::mat4 {}) == 16);

template <typename T, typename U>
consteval auto maxAlignment(T a, U b) noexcept -> size_t
{
    return std::max(alignment(a), alignment(b));
}

template <typename T, typename U, typename... Args>
consteval auto maxAlignment(T a, U b, Args... args) noexcept -> size_t
{
    return maxAlignment(maxAlignment(a, b), args...);
}

}

#define PD_ALIGN(arg) alignas(alignment(arg {})) arg

#define PD_MAKE_ALIGNED(...)                            \
    consteval auto alignment() const noexcept -> size_t \
    {                                                   \
        return maxAlignment(__VA_ARGS__, glm::vec4 {}); \
    }
