// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include "panda/gfx/Context.h"

#include <backends/imgui_impl_vulkan.h>
#include <fmt/format.h>
#include <vulkan/vk_platform.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda/Simulation.cuh>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_funcs.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_hpp_macros.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "Buffer.h"
#include "FrameInfo.h"
#include "Renderer.h"
#include "UboLight.h"
#include "panda/Logger.h"
#include "panda/Window.h"
#include "panda/gfx/Camera.h"
#include "panda/gfx/Descriptor.h"
#include "panda/gfx/Device.h"
#include "panda/gfx/Scene.h"
#include "panda/gfx/object/Mesh.h"
#include "panda/gfx/object/Texture.h"
#include "panda/gfx/systems/FluidParticleRenderSystem.h"
#include "panda/internal/config.h"
#include "panda/utils/Signal.h"
#include "panda/utils/Signals.h"
#include "panda/utils/format/gfx/api/ResultFormatter.h"  // NOLINT(misc-include-cleaner)
#include "systems/BoundaryParticleRenderSystem.hpp"
#include "systems/RenderSystem.h"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE  // cppcheck-suppress unknownMacro

#ifdef WIN32
    constexpr auto requiredDeviceExtensions = std::array {
        vk::KHRSwapchainExtensionName, vk::KHRPushDescriptorExtensionName, vk::KHRExternalMemoryWin32ExtensionName};
#else
    constexpr auto requiredDeviceExtensions = std::array {
        vk::KHRSwapchainExtensionName, vk::KHRPushDescriptorExtensionName, vk::KHRExternalMemoryFdExtensionName};
#endif

namespace panda::gfx
{
namespace
{

VKAPI_ATTR auto VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                         [[maybe_unused]] vk::DebugUtilsMessageTypeFlagsEXT messageType,
                                         const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                         [[maybe_unused]] void* pUserData) -> vk::Bool32
{
    switch (messageSeverity)
    {
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose:
        log::Debug("{}", pCallbackData->pMessage);
        break;
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo:
        log::Info("{}", pCallbackData->pMessage);
        break;
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning:
        log::Warning("{}", pCallbackData->pMessage);
        break;
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError:
        log::Error("{}", pCallbackData->pMessage);
        break;
    }

    return vk::False;
}

auto imGuiCallback(VkResult result) -> void
{
    shouldBe(vk::Result {result}, vk::Result::eSuccess, fmt::format("ImGui didn't succeed: ", vk::Result {result}));
}

}

Context::Context(const Window& window)
    : _instance {createInstance(window)},
      _debugMessenger {createDebugMessanger(*_instance)},
      _window {window}
{
    _surface = _window.createSurface(*_instance);
    log::Info("Created surface successfully");

    _device = std::make_unique<Device>(*_instance, _surface, requiredDeviceExtensions);

    log::Info("Created device successfully");
    log::Info("Chosen GPU: {}", std::string_view {_device->physicalDevice.getProperties().deviceName});

    VULKAN_HPP_DEFAULT_DISPATCHER.init(_device->logicalDevice);

    _renderer = std::make_unique<Renderer>(window, *_device, _surface);

    _uboFragBuffers.reserve(maxFramesInFlight);
    _uboVertBuffers.reserve(maxFramesInFlight);

    for (auto i = uint32_t {}; i < maxFramesInFlight; i++)
    {
        _uboFragBuffers.push_back(std::make_unique<Buffer>(
            *_device,
            sizeof(FragUbo),
            1,
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            _device->physicalDevice.getProperties().limits.minUniformBufferOffsetAlignment));
        _uboFragBuffers.back()->mapWhole();

        _uboVertBuffers.push_back(std::make_unique<Buffer>(
            *_device,
            sizeof(VertUbo),
            1,
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            _device->physicalDevice.getProperties().limits.minUniformBufferOffsetAlignment));
        _uboVertBuffers.back()->mapWhole();
    }

    _renderSystem = std::make_unique<RenderSystem>(*_device, _renderer->getSwapChainRenderPass());

    log::Info("Vulkan API has been successfully initialized");

    initializeImGui();
}

Context::~Context() noexcept
{
    log::Info("Starting closing Vulkan API");

    shouldBe(_device->logicalDevice.waitIdle(), vk::Result::eSuccess, "Wait idle didn't succeed");

    ImGui_ImplVulkan_Shutdown();

    if (_debugMessenger)
    {
        _instance->destroyDebugUtilsMessengerEXT(_debugMessenger.value());
    }
}

auto Context::initializeParticleSystem(size_t particleCount) -> sph::cuda::FluidParticlesDataImportedBuffer
{
    _particleRenderSystem =
        std::make_unique<FluidParticleRenderSystem>(*_device, _renderer->getSwapChainRenderPass(), particleCount);

    return _particleRenderSystem->getImportedMemory();
}

auto Context::initializeBoundaryParticleSystem(size_t particleCount, bool shouldRender)
    -> sph::cuda::BoundaryParticlesDataImportedBuffer
{
    _boundaryParticleRenderSystem = std::make_unique<BoundaryParticleRenderSystem>(*_device,
                                                                                   _renderer->getSwapChainRenderPass(),
                                                                                   particleCount,
                                                                                   shouldRender);

    return _boundaryParticleRenderSystem->getImportedMemory();
}

auto Context::createInstance(const Window& window) -> std::unique_ptr<vk::Instance, InstanceDeleter>
{
    const auto dynamicLoader = vk::detail::DynamicLoader {};
    const auto vkGetInstanceProcAddr = dynamicLoader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    const auto projectName = std::string {config::projectName};
    const auto engineName = std::string {config::engineName};

    const auto appInfo = vk::ApplicationInfo {.pApplicationName = projectName.data(),
                                              .applicationVersion = vk::ApiVersion10,
                                              .pEngineName = engineName.data(),
                                              .engineVersion = vk::ApiVersion10,
                                              .apiVersion = vk::ApiVersion14};

    const auto requiredExtensions = getRequiredExtensions(window);

    expect(areRequiredExtensionsAvailable(requiredExtensions), true, "There are missing extensions");

    auto createInfo = vk::InstanceCreateInfo {.pApplicationInfo = &appInfo,
                                              .enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
                                              .ppEnabledExtensionNames = requiredExtensions.data()};

    if constexpr (shouldEnableValidationLayers())
    {
        shouldBe(enableValidationLayers(createInfo), true, "Unable to enable validation layers");
        createInfo.pNext = &debugMessengerCreateInfo;
    }

    auto instance = std::unique_ptr<vk::Instance, InstanceDeleter> {
        new vk::Instance {
            expect(vk::createInstance(createInfo), vk::Result::eSuccess, "Creating instance didn't succeed")},
        InstanceDeleter {_surface}};
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);
    return instance;
}

auto Context::enableValidationLayers(vk::InstanceCreateInfo& createInfo) -> bool
{
    _requiredValidationLayers.push_back("VK_LAYER_KHRONOS_validation");

    if (areValidationLayersSupported())
    {
        createInfo.setPEnabledLayerNames(_requiredValidationLayers);

        return true;
    }
    return false;
}

auto Context::getRequiredExtensions(const Window& window) -> std::vector<const char*>
{
    auto extensions = window.getRequiredExtensions();

    if constexpr (shouldEnableValidationLayers())
    {
        extensions.push_back(vk::EXTDebugUtilsExtensionName);
    }
    extensions.push_back(vk::KHRPortabilityEnumerationExtensionName);

    return extensions;
}

auto Context::areRequiredExtensionsAvailable(std::span<const char* const> requiredExtensions) -> bool
{
    const auto availableExtensions = vk::enumerateInstanceExtensionProperties();
    if (availableExtensions.result != vk::Result::eSuccess)
    {
        log::Error("Can't get available extensions: {}", availableExtensions.result);
        return false;
    }
    for (const auto* requiredExtension : requiredExtensions)
    {
        const auto it = std::ranges::find_if(
            availableExtensions.value,
            [requiredExtension](const auto& availableExtension) {
                return std::string_view {requiredExtension} == std::string_view {availableExtension};
            },
            &vk::ExtensionProperties::extensionName);

        if (it == availableExtensions.value.cend())
        {
            log::Error("{} extension is unavailable", requiredExtension);
            return false;
        }
    }
    return true;
}

auto Context::createDebugMessanger(vk::Instance instance) -> std::optional<vk::DebugUtilsMessengerEXT>
{
    if constexpr (shouldEnableValidationLayers())
    {
        const auto debugMessenger = expect(instance.createDebugUtilsMessengerEXT(debugMessengerCreateInfo),
                                           vk::Result::eSuccess,
                                           "Unable to create debug messenger");
        log::Info("Debug messenger is created");
        return debugMessenger;
    }
    else
    {
        return std::nullopt;
    }
}

auto Context::createDebugMessengerCreateInfo() noexcept -> vk::DebugUtilsMessengerCreateInfoEXT
{
    static constexpr auto severityMask = vk::DebugUtilsMessageSeverityFlagBitsEXT::eError      //
                                         | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning  //
                                         | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo     //
                                         | vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose;

    static constexpr auto typeMask = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral       //
                                     | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation  //
                                     | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
    return {.messageSeverity = severityMask, .messageType = typeMask, .pfnUserCallback = debugCallback};
}

auto Context::areValidationLayersSupported() const -> bool
{
    const auto availableLayers = vk::enumerateInstanceLayerProperties();
    if (availableLayers.result != vk::Result::eSuccess)
    {
        log::Error("Can't enumerate available layers");
        return false;
    }

    for (const auto* layerName : _requiredValidationLayers)
    {
        const auto it = std::ranges::find_if(
            availableLayers.value,
            [layerName](const auto& availableLayer) {
                return std::string_view {layerName} == std::string_view {availableLayer};
            },
            &vk::LayerProperties::layerName);

        if (it == availableLayers.value.cend())
        {
            log::Warning("{} layer is not supported", layerName);
            return false;
        }
    }

    return true;
}

auto Context::makeFrame(Scene& scene) const -> void
{
    const auto commandBuffer = _renderer->beginFrame();
    if (!commandBuffer)
    {
        return;
    }

    const auto frameIndex = _renderer->getFrameIndex();

    const auto vertUbo = VertUbo {.projection = scene.getCamera().getProjection(), .view = scene.getCamera().getView()};

    static constexpr auto ambientColor = 0.1F;

    auto fragUbo = FragUbo {
        .inverseView = scene.getCamera().getInverseView(),
        .pointLights = {},
        .directionalLights = {},
        .spotLights = {},
        .ambientColor = {ambientColor, ambientColor, ambientColor},
        .activePointLights = {},
        .activeDirectionalLights = {},
        .activeSpotLights = {}
    };
    for (auto i = size_t {}; i < fragUbo.directionalLights.size() && i < scene.getLights().directionalLights.size();
         i++)
    {
        fragUbo.directionalLights[i] = fromDirectionalLight(scene.getLights().directionalLights[i]);
    }
    for (auto i = size_t {}; i < fragUbo.pointLights.size() && i < scene.getLights().pointLights.size(); i++)
    {
        fragUbo.pointLights[i] = fromPointLight(scene.getLights().pointLights[i]);
    }
    for (auto i = size_t {}; i < fragUbo.spotLights.size() && i < scene.getLights().spotLights.size(); i++)
    {
        fragUbo.spotLights[i] = fromSpotLight(scene.getLights().spotLights[i]);
    }

    fragUbo.activeDirectionalLights = static_cast<uint32_t>(scene.getLights().directionalLights.size());
    fragUbo.activePointLights = static_cast<uint32_t>(scene.getLights().pointLights.size());
    fragUbo.activeSpotLights = static_cast<uint32_t>(scene.getLights().spotLights.size());

    _uboVertBuffers[frameIndex]->writeAt(vertUbo, 0);
    _uboFragBuffers[frameIndex]->writeAt(fragUbo, 0);
    _renderer->beginSwapChainRenderPass();

    if (_renderSystem != nullptr) [[likely]]
    {
        _renderSystem->render(FrameInfo {.scene = scene,
                                         .fragUbo = *_uboFragBuffers[frameIndex],
                                         .vertUbo = *_uboVertBuffers[frameIndex],
                                         .commandBuffer = commandBuffer,
                                         .frameIndex = frameIndex});
    }

    if (_particleRenderSystem != nullptr) [[likely]]
    {
        _particleRenderSystem->render(FrameInfo {.scene = scene,
                                                 .fragUbo = *_uboFragBuffers[frameIndex],
                                                 .vertUbo = *_uboVertBuffers[frameIndex],
                                                 .commandBuffer = commandBuffer,
                                                 .frameIndex = frameIndex});
    }

    if (_boundaryParticleRenderSystem != nullptr)
    {
        _boundaryParticleRenderSystem->render(FrameInfo {.scene = scene,
                                                         .fragUbo = *_uboFragBuffers[frameIndex],
                                                         .vertUbo = *_uboVertBuffers[frameIndex],
                                                         .commandBuffer = commandBuffer,
                                                         .frameIndex = frameIndex});
    }

    utils::signals::beginGuiRender.registerSender()(
        utils::signals::BeginGuiRenderData {.commandBuffer = commandBuffer, .scene = std::ref(scene)});

    _renderer->endSwapChainRenderPass();

    _renderer->endFrame();
}

auto Context::getDevice() const noexcept -> const Device&
{
    return *_device;
}

auto Context::initializeImGui() -> void
{
    _guiPool = DescriptorPool::Builder(*_device)
                   .addPoolSize(vk::DescriptorType::eCombinedImageSampler, maxFramesInFlight)
                   .build(maxFramesInFlight, vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
    auto initInfo = ImGui_ImplVulkan_InitInfo {.Instance = *_instance,
                                               .PhysicalDevice = _device->physicalDevice,
                                               .Device = _device->logicalDevice,
                                               .QueueFamily = _device->queueFamilies.graphicsFamily,
                                               .Queue = _device->graphicsQueue,
                                               .DescriptorPool = _guiPool->getHandle(),
                                               .RenderPass = _renderer->getSwapChainRenderPass(),
                                               .MinImageCount = maxFramesInFlight,
                                               .ImageCount = maxFramesInFlight,
                                               .MSAASamples = VK_SAMPLE_COUNT_1_BIT,
                                               .PipelineCache = {},
                                               .Subpass = {},
                                               .DescriptorPoolSize = {},
                                               .UseDynamicRendering = false,
                                               .PipelineRenderingCreateInfo = {},
                                               .Allocator = {},
                                               .CheckVkResultFn = imGuiCallback,
                                               .MinAllocationSize = {}};

    ImGui_ImplVulkan_Init(&initInfo);
}

auto Context::registerMesh(std::unique_ptr<Mesh> mesh) -> void
{
    _meshes.push_back(std::move(mesh));
}

auto Context::getAspectRatio() const noexcept -> float
{
    return _renderer->getAspectRatio();
}

auto Context::registerTexture(std::unique_ptr<Texture> texture) -> void
{
    _textures.push_back(std::move(texture));
}

auto Context::InstanceDeleter::operator()(vk::Instance* instance) const noexcept -> void
{
    log::Info("Destroying instance");
    instance->destroy(surface);
    instance->destroy();

    delete instance;  //NOLINT(cppcoreguidelines-owning-memory)
}
}
