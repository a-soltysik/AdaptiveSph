#include "App.hpp"

#include <GLFW/glfw3.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <fmt/format.h>
#include <imgui.h>
#include <panda/Logger.h>
#include <panda/gfx/Camera.h>
#include <panda/gfx/Light.h>
#include <panda/gfx/vulkan/object/Object.h>
#include <panda/gfx/vulkan/object/Surface.h>
#include <panda/gfx/vulkan/object/Texture.h>
#include <panda/utils/Assert.h>
#include <panda/utils/Signals.h>
#include <vulkan/vulkan_core.h>

#include <array>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <glm/common.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_uint2.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/trigonometric.hpp>
#include <input_handler/MouseHandler.hpp>
#include <memory>
#include <mesh/InvertedCube.hpp>
#include <mesh/UvSphere.hpp>
#include <string_view>
#include <utility>

#include "internal/config.hpp"
#include "movement_handler/MovementHandler.hpp"
#include "movement_handler/RotationHandler.hpp"

namespace
{

[[nodiscard]] constexpr auto getSignalName(const int signalValue) noexcept -> std::string_view
{
    switch (signalValue)
    {
    case SIGABRT:
        return "SIGABRT";
    case SIGFPE:
        return "SIGFPE";
    case SIGILL:
        return "SIGILL";
    case SIGINT:
        return "SIGINT";
    case SIGSEGV:
        return "SIGSEGV";
    case SIGTERM:
        return "SIGTERM";
    default:
        return "unknown";
    }
}

[[noreturn]] auto signalHandler(const int signalValue) -> void
{
    panda::log::Error("Received {} signal", getSignalName(signalValue));
    panda::log::FileLogger::instance().terminate();
    std::_Exit(signalValue);
}

void processCamera(const float deltaTime,
                   const sph::Window& window,
                   panda::gfx::vulkan::Transform& cameraObject,
                   panda::gfx::Camera& camera)
{
    static constexpr auto rotationVelocity = 500.F;
    static constexpr auto moveVelocity = 2.5F;

    if (window.getMouseHandler().getButtonState(GLFW_MOUSE_BUTTON_LEFT) == sph::MouseHandler::ButtonState::Pressed)
    {
        cameraObject.rotation +=
            glm::vec3 {sph::RotationHandler {window}.getRotation() * rotationVelocity * deltaTime, 0};
    }

    cameraObject.rotation.x = glm::clamp(cameraObject.rotation.x, -glm::half_pi<float>(), glm::half_pi<float>());
    cameraObject.rotation.y = glm::mod(cameraObject.rotation.y, glm::two_pi<float>());

    const auto rawMovement = sph::MovementHandler {window}.getMovement();

    const auto cameraDirection = glm::vec3 {glm::cos(-cameraObject.rotation.x) * glm::sin(cameraObject.rotation.y),
                                            glm::sin(-cameraObject.rotation.x),
                                            glm::cos(-cameraObject.rotation.x) * glm::cos(cameraObject.rotation.y)};
    const auto cameraRight = glm::vec3 {glm::cos(cameraObject.rotation.y), 0, -glm::sin(cameraObject.rotation.y)};

    auto translation = cameraDirection * rawMovement.z.value_or(0);
    translation += cameraRight * rawMovement.x.value_or(0);
    translation.y = rawMovement.y.value_or(translation.y);

    if (glm::dot(translation, translation) > 0)
    {
        cameraObject.translation += glm::normalize(translation) * moveVelocity * deltaTime;
    }

    camera.setViewYXZ(panda::gfx::view::YXZ {
        .position = cameraObject.translation,
        .rotation = {-cameraObject.rotation.x, cameraObject.rotation.y, 0}
    });
}

class TimeData
{
public:
    TimeData()
        : _time {std::chrono::steady_clock::now()}
    {
    }

    auto update() -> void
    {
        const auto currentTime = std::chrono::steady_clock::now();
        _deltaTime = std::chrono::duration<float>(currentTime - _time).count();
        _time = currentTime;
    }

    [[nodiscard]] auto getDelta() const noexcept -> float
    {
        return _deltaTime;
    }

private:
    std::chrono::steady_clock::time_point _time;
    float _deltaTime {};
};
}

namespace sph
{

auto App::run() -> int
{
    initializeLogger();
    registerSignalHandlers();

    static constexpr auto defaultWidth = uint32_t {1920};
    static constexpr auto defaultHeight = uint32_t {1080};
    _window = std::make_unique<Window>(glm::uvec2 {defaultWidth, defaultHeight}, config::project_name.data());
    _api = std::make_unique<panda::gfx::vulkan::Context>(*_window, 250000, false);

    setDefaultScene();
    mainLoop();
    return 0;
}

auto App::mainLoop() -> void
{
    auto currentTime = TimeData {};

    auto cameraObject = panda::gfx::vulkan::Transform {};

    cameraObject.translation = {0, 0.5, -5};
    _scene.getCamera().setViewYXZ(
        panda::gfx::view::YXZ {.position = cameraObject.translation, .rotation = cameraObject.rotation});

    const auto beginGuiReceiver = panda::utils::signals::beginGuiRender.connect([](auto data) {
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        showFpsOverlay();
        ImGui::Render();

        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), data.commandBuffer, VK_NULL_HANDLE);
    });

    while (!_window->shouldClose()) [[likely]]
    {
        if (!_window->isMinimized())
        {
            panda::utils::signals::gameLoopIterationStarted.registerSender()();
            _window->processInput();

            currentTime.update();

            _scene.getCamera().setPerspectiveProjection(
                panda::gfx::projection::Perspective {.fovY = glm::radians(50.F),
                                                     .aspect = _api->getRenderer().getAspectRatio(),
                                                     .near = 0.1F,
                                                     .far = 100});
            processCamera(currentTime.getDelta(), *_window, cameraObject, _scene.getCamera());

            _api->makeFrame(currentTime.getDelta(), _scene);
        }
        else [[unlikely]]
        {
            _window->waitForInput();
        }
    }
}

auto App::initializeLogger() -> void
{
    using enum panda::log::Level;

    if constexpr (config::isDebug)
    {
        panda::log::FileLogger::instance().setLevels(std::array {Debug, Info, Warning, Error});
    }
    else
    {
        panda::log::FileLogger::instance().setLevels(std::array {Info, Warning, Error});
    }

    panda::log::FileLogger::instance().start();
}

auto App::registerSignalHandlers() -> void
{
    panda::shouldNotBe(std::signal(SIGABRT, signalHandler), SIG_ERR, "Failed to register signal handler");
    panda::shouldNotBe(std::signal(SIGFPE, signalHandler), SIG_ERR, "Failed to register signal handler");
    panda::shouldNotBe(std::signal(SIGILL, signalHandler), SIG_ERR, "Failed to register signal handler");
    panda::shouldNotBe(std::signal(SIGINT, signalHandler), SIG_ERR, "Failed to register signal handler");
    panda::shouldNotBe(std::signal(SIGSEGV, signalHandler), SIG_ERR, "Failed to register signal handler");
    panda::shouldNotBe(std::signal(SIGTERM, signalHandler), SIG_ERR, "Failed to register signal handler");
}

void App::setDefaultScene()
{
    auto redTexture = panda::gfx::vulkan::Texture::getDefaultTexture(*_api, {1, 0, 0, 1});
    auto blueTexture = panda::gfx::vulkan::Texture::getDefaultTexture(*_api, {0, 0, 1, 1});
    auto sphereMesh = mesh::uv_sphere::create(*_api, "Sphere", {.radius = 1, .stacks = 5, .slices = 10});
    auto invertedCubeMesh = mesh::inverted_cube::create(*_api, "InvertedCube");

    auto& object = _scene.addObject("Domain",
                                    {
                                        panda::gfx::vulkan::Surface {blueTexture.get(), invertedCubeMesh.get(), true}
    });
    object.transform.rotation = {0, 0, 0};
    object.transform.translation = {0, 0.F, 0};
    object.transform.scale = {5, 2, 3};

    auto index = 0;
    for (auto i = 0; i < 100; i++)
    {
        for (auto j = 0; j < 60; j++)
        {
            for (auto k = 0; k < 40; k++)
            {
                auto& sphere =
                    _scene.addObject(fmt::format("Sphere#{}", index++),
                                     {
                                         panda::gfx::vulkan::Surface {redTexture.get(), sphereMesh.get(), true}
                });
                sphere.transform.scale = {0.025, 0.025, 0.025};
                sphere.transform.translation = {
                    (-object.transform.scale.x / 2) + sphere.transform.scale.x + (static_cast<float>(i) / 20),
                    (-object.transform.scale.y / 2) + sphere.transform.scale.y + (static_cast<float>(k) / 20),
                    (-object.transform.scale.z / 2) + sphere.transform.scale.z + (static_cast<float>(j) / 20)};
            }
        }
    }

    _api->registerMesh(std::move(sphereMesh));
    _api->registerMesh(std::move(invertedCubeMesh));
    _api->registerTexture(std::move(redTexture));
    _api->registerTexture(std::move(blueTexture));

    auto directionalLight = _scene.addLight<panda::gfx::DirectionalLight>("DirectionalLight");

    if (directionalLight.has_value())
    {
        directionalLight.value().get().makeColorLight({1.F, .8F, .8F}, 0.F, 0.8F, 1.F, 0.8F);
        directionalLight.value().get().direction = {-6.2F, -2.F, 0};
    }

    directionalLight = _scene.addLight<panda::gfx::DirectionalLight>("DirectionalLight#1");

    if (directionalLight.has_value())
    {
        directionalLight.value().get().makeColorLight({1.F, .8F, .8F}, 0.F, 0.8F, 1.F, 0.8F);
        directionalLight.value().get().direction = {6.2F, 2.F, 0};
    }
}

auto App::showFpsOverlay() -> void
{
    static float lastFps = 0.0F;
    static auto lastUpdate = std::chrono::steady_clock::now();

    const auto now = std::chrono::steady_clock::now();
    const auto deltaTime = std::chrono::duration<float>(now - lastUpdate).count();

    if (deltaTime >= 1.0F)
    {
        lastFps = ImGui::GetIO().Framerate;
        lastUpdate = now;
    }

    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);

    if (ImGui::Begin("FPS Overlay",
                     nullptr,
                     static_cast<uint32_t>(ImGuiWindowFlags_NoDecoration) |
                         static_cast<uint32_t>(ImGuiWindowFlags_AlwaysAutoResize) |
                         static_cast<uint32_t>(ImGuiWindowFlags_NoSavedSettings) |
                         static_cast<uint32_t>(ImGuiWindowFlags_NoFocusOnAppearing) |
                         static_cast<uint32_t>(ImGuiWindowFlags_NoNav)))
    {
        //NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
        ImGui::Text("FPS: %.1f", static_cast<double>(lastFps));
    }
    ImGui::End();
}

}
