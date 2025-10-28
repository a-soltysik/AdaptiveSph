#pragma once

// clang-format off
#include "panda/utils/Assert.h" // NOLINT(misc-include-cleaner)
// clang-format on

#include <cstddef>
#include <memory>
#include <span>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "cuda/Simulation.cuh"
#include "panda/Common.h"
#include "panda/internal/config.h"

namespace panda
{

class Window;

namespace gfx
{

class Buffer;
class DescriptorPool;
class Device;
class Renderer;
class Scene;
class Mesh;
class Texture;
class ParticleRenderSystem;
class RenderSystem;

class Context
{
public:
    explicit Context(const Window& window);
    PD_DELETE_ALL(Context);
    ~Context() noexcept;

    static constexpr auto maxFramesInFlight = size_t {2};

    auto initializeParticleSystem(size_t particleCount) -> sph::cuda::ParticlesDataBuffer;
    auto makeFrame(Scene& scene) const -> void;
    [[nodiscard]] auto getDevice() const noexcept -> const Device&;
    auto registerTexture(std::unique_ptr<Texture> texture) -> void;
    auto registerMesh(std::unique_ptr<Mesh> mesh) -> void;
    [[nodiscard]] auto getAspectRatio() const noexcept -> float;

private:
    struct InstanceDeleter
    {
        auto operator()(vk::Instance* instance) const noexcept -> void;
        const vk::SurfaceKHR& surface;
    };

    [[nodiscard]] static constexpr auto shouldEnableValidationLayers() noexcept -> bool
    {
        return config::isDebug;
    }

    [[nodiscard]] static auto getRequiredExtensions(const Window& window) -> std::vector<const char*>;
    [[nodiscard]] auto createInstance(const Window& window) -> std::unique_ptr<vk::Instance, InstanceDeleter>;
    [[nodiscard]] static auto createDebugMessengerCreateInfo() noexcept -> vk::DebugUtilsMessengerCreateInfoEXT;
    [[nodiscard]] static auto areRequiredExtensionsAvailable(std::span<const char* const> requiredExtensions) -> bool;
    [[nodiscard]] static auto createDebugMessanger(vk::Instance instance) -> std::optional<vk::DebugUtilsMessengerEXT>;

    [[nodiscard]] auto areValidationLayersSupported() const -> bool;

    auto enableValidationLayers(vk::InstanceCreateInfo& createInfo) -> bool;
    auto initializeImGui() -> void;

    inline static const vk::DebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo =
        createDebugMessengerCreateInfo();

    vk::SurfaceKHR _surface;
    std::vector<const char*> _requiredValidationLayers;
    std::unique_ptr<vk::Instance, InstanceDeleter> _instance;
    std::unique_ptr<Device> _device;
    std::unique_ptr<Renderer> _renderer;
    std::unique_ptr<RenderSystem> _renderSystem;
    std::unique_ptr<ParticleRenderSystem> _particleRenderSystem;
    std::optional<vk::DebugUtilsMessengerEXT> _debugMessenger;
    std::vector<std::unique_ptr<Texture>> _textures;
    std::vector<std::unique_ptr<Mesh>> _meshes;
    std::vector<std::unique_ptr<Buffer>> _uboFragBuffers;
    std::vector<std::unique_ptr<Buffer>> _uboVertBuffers;
    std::unique_ptr<DescriptorPool> _guiPool;

    const Window& _window;
};

}
}