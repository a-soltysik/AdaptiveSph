#include "Window.hpp"

#include <GLFW/glfw3.h>
#include <backends/imgui_impl_glfw.h>
#include <imgui.h>
#include <panda/Logger.h>
#include <panda/Window.h>
#include <panda/utils/Assert.h>
#include <panda/utils/Signals.h>
#include <vulkan/vulkan_core.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <glm/ext/vector_uint2.hpp>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "../input/KeyboardHandler.hpp"
#include "../input/MouseHandler.hpp"

namespace
{

void framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
    static const auto sender = panda::utils::signals::frameBufferResized.registerSender();
    const auto windowId = sph::Window::makeId(window);
    panda::log::Info("Size of window [{}] changed to {}x{}", windowId, width, height);

    sender(panda::utils::signals::FrameBufferResizedData {.id = windowId, .x = width, .y = height});
}

}

namespace sph
{

Window::Window(const glm::uvec2 size, const char* name)
    : _window {createWindow(size, name)},
      _size {size}
{
    _keyboardHandler = std::make_unique<KeyboardHandler>(*this);
    _mouseHandler = std::make_unique<MouseHandler>(*this);
    glfwSetFramebufferSizeCallback(_window, framebufferResizeCallback);

    _frameBufferResizedReceiver = panda::utils::signals::frameBufferResized.connect([this](auto data) {
        if (data.id == getId())
        {
            panda::log::Debug("Received framebuffer resized notif");
            _size = {data.x, data.y};
        }
    });

    setupImGui();
}

Window::Window(Window&& rhs) noexcept
    : _keyboardHandler {std::move(rhs._keyboardHandler)},
      _mouseHandler {std::move(rhs._mouseHandler)},
      _frameBufferResizedReceiver {std::move(rhs._frameBufferResizedReceiver)},
      _window {nullptr},
      _size {rhs._size}
{
}

Window::~Window() noexcept
{
    if (_window == nullptr)
    {
        return;
    }
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(_window);
    glfwTerminate();
    panda::log::Info("Window [{}] destroyed", static_cast<void*>(_window));
}

auto Window::createWindow(glm::uvec2 size, const char* name) -> GLFWwindow*
{
    panda::expect(glfwInit(), GLFW_TRUE, "Failed to initialize GLFW");

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    auto* window = glfwCreateWindow(static_cast<int>(size.x), static_cast<int>(size.y), name, nullptr, nullptr);
    panda::log::Info("Window [{}] {}x{} px created", static_cast<void*>(window), size.x, size.y);

    return window;
}

auto Window::getRequiredExtensions() const -> std::vector<const char*>
{
    auto glfwExtensionsCount = uint32_t {};
    const auto* glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionsCount);

    if (glfwExtensions == nullptr)
    {
        return {};
    }

    const auto extensionsSpan = std::span(glfwExtensions, glfwExtensionsCount);

    return {extensionsSpan.begin(), extensionsSpan.end()};
}

auto Window::createSurface(VkInstance instance) const -> VkSurfaceKHR
{
    auto* newSurface = VkSurfaceKHR {};
    glfwCreateWindowSurface(instance, _window, nullptr, &newSurface);

    return panda::expect(
        newSurface,
        [](const auto* result) {
            return result != nullptr;
        },
        "Unable to create surface");
}

auto Window::shouldClose() const -> bool
{
    return glfwWindowShouldClose(_window) == GLFW_TRUE;
}

auto Window::processInput() -> void
{
    glfwPollEvents();
}

auto Window::getSize() const -> glm::uvec2
{
    return _size;
}

auto Window::isMinimized() const -> bool
{
    return _size.x == 0 || _size.y == 0;
}

auto Window::waitForInput() -> void
{
    glfwWaitEvents();
}

auto Window::getId() const -> size_t
{
    return makeId(_window);
}

auto Window::makeId(GLFWwindow* window) -> panda::Window::Id
{
    return std::bit_cast<size_t>(window);
}

auto Window::setupImGui() const -> void
{
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForVulkan(_window, true);
}

auto Window::setKeyCallback(GLFWkeyfun callback) const noexcept -> GLFWkeyfun
{
    return glfwSetKeyCallback(_window, callback);
}

auto Window::setMouseButtonCallback(GLFWmousebuttonfun callback) const noexcept -> GLFWmousebuttonfun
{
    return glfwSetMouseButtonCallback(_window, callback);
}

auto Window::setCursorPositionCallback(GLFWcursorposfun callback) const noexcept -> GLFWcursorposfun
{
    return glfwSetCursorPosCallback(_window, callback);
}

auto Window::getKeyboardHandler() const noexcept -> const KeyboardHandler&
{
    return *_keyboardHandler;
}

auto Window::getMouseHandler() const noexcept -> const MouseHandler&
{
    return *_mouseHandler;
}

}
