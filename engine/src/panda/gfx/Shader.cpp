// clang-format off
#include "panda/utils/Assert.h" // NOLINT(misc-include-cleaner)
// clang-format on

#include "panda/gfx/Shader.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "panda/Logger.h"
#include "panda/utils/format/gfx/api/ResultFormatter.h"  // NOLINT(misc-include-cleaner)

namespace panda::gfx
{

auto Shader::createFromFile(const vk::Device& device, const std::filesystem::path& path) -> std::optional<Shader>
{
    using namespace std::string_view_literals;
    static const std::unordered_map<std::string_view, Type> extensions = {
        {".vert"sv, Type::Vertex                },
        {".tesc"sv, Type::TessellationControl   },
        {".tese"sv, Type::TessellationEvaluation},
        {".geom"sv, Type::Geometry              },
        {".frag"sv, Type::Fragment              },
        {".comp"sv, Type::Compute               }
    };

    const auto pathStr = path.string();
    const auto lastExtension = pathStr.rfind('.');
    const auto shaderExtension =
        pathStr.substr(pathStr.rfind('.', lastExtension - 1), pathStr.size() - lastExtension + 1);

    const auto it = extensions.find(shaderExtension);
    if (it == extensions.cend())
    {
        log::Warning("File extension: {} is not supported (filename: {})", shaderExtension, pathStr);
        return {};
    }
    return createFromFile(device, path, it->second);
}

auto Shader::createFromFile(const vk::Device& device, const std::filesystem::path& path, Type type)
    -> std::optional<Shader>
{
    auto fin = std::ifstream(path, std::ios::ate | std::ios::binary);

    if (!fin.is_open())
    {
        log::Warning("File {} cannot be opened", path.string());
        return {};
    }

    const auto fileSize = fin.tellg();
    const auto bufferSize = static_cast<size_t>(fileSize) / sizeof(uint32_t);
    auto buffer = std::vector<uint32_t>(bufferSize);

    fin.seekg(0);
    fin.read(reinterpret_cast<char*>(buffer.data()), fileSize);

    return createFromRawData(device, buffer, type);
}

auto Shader::createFromRawData(const vk::Device& device, const std::vector<uint32_t>& buffer, Type type)
    -> std::optional<Shader>
{
    const auto createInfo =
        vk::ShaderModuleCreateInfo {.codeSize = buffer.size() * sizeof(uint32_t), .pCode = buffer.data()};
    const auto shaderModuleResult = device.createShaderModule(createInfo);
    if (shaderModuleResult.result == vk::Result::eSuccess)
    {
        return std::make_optional<Shader>(shaderModuleResult.value, type, device);
    }

    log::Warning("Creating shader module didn't succeed: {}", shaderModuleResult.result);
    return {};
}

Shader::~Shader() noexcept
{
    log::Info("Destroying shader [{}]", static_cast<void*>(module));
    _device.destroy(module);
}

Shader::Shader(const vk::ShaderModule& shaderModule, Type shaderType, const vk::Device& device) noexcept
    : module {shaderModule},
      type {shaderType},
      _device {device}
{
    log::Info("Created shader [{}]", static_cast<void*>(module));
}

}