#pragma once

// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include <filesystem>
#include <glm/ext/vector_float4.hpp>
#include <memory>
#include <span>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

namespace panda::gfx
{

class Context;

class Texture
{
public:
    [[nodiscard]] static auto getDefaultTexture(const Context& context, glm::vec4 color = {1.F, 1.F, 1.F, 1.F})
        -> std::unique_ptr<Texture>;
    Texture(const Context& context, std::span<const uint8_t> data, size_t width, size_t height);
    PD_DELETE_ALL(Texture);
    ~Texture();

    [[nodiscard]] auto getDescriptorImageInfo() const noexcept -> vk::DescriptorImageInfo;

private:
    auto load(std::span<const uint8_t> data, size_t width, size_t height) -> void;

    const Context& _context;
    vk::Image _image;
    vk::ImageView _imageView;
    vk::DeviceMemory _imageMemory;
    vk::Sampler _sampler;
};

}
