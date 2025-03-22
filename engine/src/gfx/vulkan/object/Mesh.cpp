// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include "panda/gfx/vulkan/object/Mesh.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "panda/Logger.h"
#include "panda/gfx/vulkan/Buffer.h"
#include "panda/gfx/vulkan/Device.h"
#include "panda/gfx/vulkan/Vertex.h"

namespace panda::gfx::vulkan
{

Mesh::Mesh(std::string name, const Device& device, std::span<const Vertex> vertices, std::span<const uint32_t> indices)
    : _device {device},
      _name {std::move(name)},
      _vertexBuffer {createVertexBuffer(_device, vertices)},
      _indexBuffer {createIndexBuffer(_device, indices)},
      _vertexCount {static_cast<uint32_t>(vertices.size())},
      _indexCount {static_cast<uint32_t>(indices.size())}
{
    log::Info("Created Mesh with {} vertices and {} indices", _vertexCount, _indexCount);
}

auto Mesh::createVertexBuffer(const Device& device, std::span<const Vertex> vertices) -> std::unique_ptr<Buffer>
{
    expect(
        vertices.size(),
        [](const auto size) {
            return size >= 3;
        },
        "Vertices size should be greater or equal to 3");

    const auto stagingBuffer =
        Buffer {device,
                vertices,
                vk::BufferUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent};

    auto newVertexBuffer =
        std::make_unique<Buffer>(device,
                                 stagingBuffer.size,
                                 vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal);

    Buffer::copy(stagingBuffer, *newVertexBuffer);

    return newVertexBuffer;
}

auto Mesh::bind(const vk::CommandBuffer& commandBuffer) const -> void
{
    commandBuffer.bindVertexBuffers(0, _vertexBuffer->buffer, {0});

    if (_indexBuffer != nullptr)
    {
        commandBuffer.bindIndexBuffer(_indexBuffer->buffer, 0, vk::IndexType::eUint32);
    }
}

auto Mesh::draw(const vk::CommandBuffer& commandBuffer) const -> void
{
    if (_indexBuffer != nullptr)
    {
        commandBuffer.drawIndexed(_indexCount, 1, 0, 0, 0);
    }
    else
    {
        commandBuffer.draw(_vertexCount, 1, 0, 0);
    }
}

auto Mesh::drawInstanced(const vk::CommandBuffer& commandBuffer, uint32_t instanced, uint32_t base) const -> void
{
    if (_indexBuffer != nullptr)
    {
        commandBuffer.drawIndexed(_indexCount, instanced, 0, 0, base);
    }
    else
    {
        commandBuffer.draw(_vertexCount, instanced, 0, base);
    }
}

auto Mesh::createIndexBuffer(const Device& device, const std::span<const uint32_t> indices) -> std::unique_ptr<Buffer>
{
    if (indices.empty())
    {
        return nullptr;
    }

    if (!shouldBe(
            indices.size(),
            [](const auto size) {
                return size >= 3;
            },
            "Indices  size should be greater or equal to 3"))
    {
        return nullptr;
    }

    const auto stagingBuffer =
        Buffer {device,
                indices,
                vk::BufferUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent};

    auto newIndexBuffer =
        std::make_unique<Buffer>(device,
                                 stagingBuffer.size,
                                 vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal);

    Buffer::copy(stagingBuffer, *newIndexBuffer);

    return newIndexBuffer;
}

auto Mesh::getName() const noexcept -> const std::string&
{
    return _name;
}

}
