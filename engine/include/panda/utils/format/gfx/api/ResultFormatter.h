#pragma once

// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include <fmt/base.h>
#include <fmt/format.h>

#include <string_view>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_to_string.hpp>

template <>
struct fmt::formatter<vk::Result> : formatter<std::string_view>
{
    template <typename FormatContext>
    [[nodiscard]] auto format(vk::Result result, FormatContext& ctx) const -> decltype(ctx.out())
    {
        return formatter<std::string_view>::format(vk::to_string(result), ctx);
    }
};