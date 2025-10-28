#pragma once

// clang-format off
#include "panda/utils/Assert.h" // NOLINT(misc-include-cleaner)
// clang-format on

#include "panda/gfx/Vertex.h"

namespace panda::gfx
{

class Context;
class Buffer;
class Device;

class Mesh
{
public:
    Mesh(std::string name,
         const Device& device,
         std::span<const Vertex> vertices,
         std::span<const uint32_t> indices = {});
    ~Mesh();

    auto bind(const vk::CommandBuffer& commandBuffer) const -> void;
    auto draw(const vk::CommandBuffer& commandBuffer) const -> void;
    auto drawInstanced(const vk::CommandBuffer& commandBuffer, uint32_t instanced, uint32_t base) const -> void;

    [[nodiscard]] auto getName() const noexcept -> const std::string&;

private:
    static auto createVertexBuffer(const Device& device, std::span<const Vertex> vertices) -> std::unique_ptr<Buffer>;
    static auto createIndexBuffer(const Device& device, std::span<const uint32_t> indices) -> std::unique_ptr<Buffer>;

    const Device& _device;
    std::string _name;
    std::unique_ptr<Buffer> _vertexBuffer;
    std::unique_ptr<Buffer> _indexBuffer;
    uint32_t _vertexCount;
    uint32_t _indexCount;
};

}
