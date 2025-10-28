#pragma once

// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "panda/Common.h"

namespace panda::gfx
{

class Shader
{
public:
    enum class Type : uint8_t
    {
        Vertex,
        TessellationControl,
        TessellationEvaluation,
        Geometry,
        Fragment,
        Compute
    };

    Shader(const vk::ShaderModule& shaderModule, Type shaderType, const vk::Device& device) noexcept;
    PD_DELETE_ALL(Shader);
    ~Shader() noexcept;

    [[nodiscard]] static auto createFromFile(const vk::Device& device, const std::filesystem::path& path)
        -> std::optional<Shader>;
    [[nodiscard]] static auto createFromFile(const vk::Device& device, const std::filesystem::path& path, Type type)
        -> std::optional<Shader>;
    [[nodiscard]] static auto createFromRawData(const vk::Device& device,
                                                const std::vector<uint32_t>& buffer,
                                                Type type) -> std::optional<Shader>;

    [[nodiscard]] static constexpr auto getEntryPointName() -> const char*
    {
        return "main";
    }

    const vk::ShaderModule module;
    const Type type;

private:
    const vk::Device& _device;
};

}
