#include "App.hpp"

#include <GLFW/glfw3.h>
#include <panda/Logger.h>
#include <panda/gfx/Camera.h>
#include <panda/gfx/Light.h>
#include <panda/gfx/vulkan/object/Object.h>
#include <panda/gfx/vulkan/object/Surface.h>
#include <panda/gfx/vulkan/object/Texture.h>
#include <panda/utils/Assert.h>
#include <panda/utils/Signals.h>

#include <array>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cuda/Simulation.cuh>
#include <glm/common.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_uint2.hpp>
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/trigonometric.hpp>
#include <gui/SimulationDataGui.hpp>
#include <input_handler/MouseHandler.hpp>
#include <memory>
#include <mesh/UvSphere.hpp>
#include <ranges>
#include <string_view>
#include <utility>
#include <vector>

#include "glm/exponential.hpp"
#include "glm/gtx/string_cast.hpp"
#include "internal/config.hpp"
#include "mesh/InvertedCube.hpp"
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
    _api = std::make_unique<panda::gfx::vulkan::Context>(*_window);

    auto redTexture = panda::gfx::vulkan::Texture::getDefaultTexture(*_api, {1, 0, 0, 1});

    _scene = std::make_unique<panda::gfx::vulkan::Scene>(panda::gfx::vulkan::Surface {redTexture.get(), nullptr});

    _api->registerTexture(std::move(redTexture));

    setDefaultScene();

    auto translations =
        _scene->getParticles().objects | std::ranges::views::transform(&panda::gfx::vulkan::Object::transform) |
        std::ranges::views::transform(&panda::gfx::vulkan::Transform::translation) | std::ranges::views::common;

    const std::vector translationVec(translations.begin(), translations.end());

    _simulation = createSimulation(_simulationParameters,
                                   translationVec,
                                   _api->initializeParticleSystem(sizeof(cuda::ParticleData), 250000));

    mainLoop();
    return 0;
}

auto App::mainLoop() const -> void
{
    auto currentTime = TimeData {};

    auto cameraObject = panda::gfx::vulkan::Transform {};

    cameraObject.translation = {0, 0.5, -5};
    _scene->getCamera().setViewYXZ(
        panda::gfx::view::YXZ {.position = cameraObject.translation, .rotation = cameraObject.rotation});

    auto gui = SimulationDataGui {*_window, _simulationParameters};

    while (!_window->shouldClose()) [[likely]]
    {
        if (!_window->isMinimized()) [[likely]]
        {
            panda::utils::signals::gameLoopIterationStarted.registerSender()();
            _window->processInput();

            currentTime.update();
            const auto guiUpdate = gui.getParameters();
            _scene->getDomain().transform.translation = (guiUpdate.domain.max + guiUpdate.domain.min) / 2.F;
            _simulation->update(guiUpdate, currentTime.getDelta());
            _scene->getCamera().setPerspectiveProjection(
                panda::gfx::projection::Perspective {.fovY = glm::radians(50.F),
                                                     .aspect = _api->getRenderer().getAspectRatio(),
                                                     .zNear = 0.1F,
                                                     .zFar = 100});
            processCamera(currentTime.getDelta(), *_window, cameraObject, _scene->getCamera());

            _api->makeFrame(currentTime.getDelta(), *_scene);
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

    auto& object = _scene->setDomain("Domain",
                                     {
                                         panda::gfx::vulkan::Surface {blueTexture.get(), invertedCubeMesh.get()}
    });
    object.transform.rotation = {};
    object.transform.translation = {};
    object.transform.scale = {5.F, 2.F, 3.F};

    _simulationParameters = getInitialSimulationParameters(
        cuda::Simulation::Parameters::Domain {}.fromTransform(object.transform.translation, object.transform.scale),
        20000,
        1000.F);

    const auto simulationSize = glm::uvec3 {40, 20, 25};
    const auto maxSize = glm::vec3 {simulationSize} * _simulationParameters.particleRadius * 2.F * 1.5F;

    for (auto i = uint32_t {}; i < simulationSize.x; i++)
    {
        for (auto j = uint32_t {}; j < simulationSize.y; j++)
        {
            for (auto k = uint32_t {}; k < simulationSize.z; k++)
            {
                auto& sphere = _scene->addParticle();

                sphere.transform.translation =
                    -maxSize / 2.F +
                    glm::vec3 {_simulationParameters.particleRadius * 2 * 1.5F * static_cast<float>(i),
                               _simulationParameters.particleRadius * 2 * 1.5F * static_cast<float>(j),
                               _simulationParameters.particleRadius * 2 * 1.5F * static_cast<float>(k)};

                _particles.push_back(&sphere.transform.translation);
            }
        }
    }

    _api->registerMesh(std::move(sphereMesh));
    _api->registerMesh(std::move(invertedCubeMesh));
    _api->registerTexture(std::move(redTexture));
    _api->registerTexture(std::move(blueTexture));

    auto directionalLight = _scene->addLight<panda::gfx::DirectionalLight>("DirectionalLight");

    if (directionalLight.has_value())
    {
        directionalLight.value().get().makeColorLight({1.F, .8F, .8F}, 0.F, 0.8F, 1.F, 0.8F);
        directionalLight.value().get().direction = {-6.2F, -2.F, -1.F};
    }

    directionalLight = _scene->addLight<panda::gfx::DirectionalLight>("DirectionalLight#1");

    if (directionalLight.has_value())
    {
        directionalLight.value().get().makeColorLight({1.F, .8F, .8F}, 0.1F, 0.8F, 1.F, 0.8F);
        directionalLight.value().get().direction = {6.2F, 2.F, 1.F};
    }
}

auto App::getInitialSimulationParameters(const cuda::Simulation::Parameters::Domain& domain,
                                         uint32_t particleCount,
                                         float totalMass) -> cuda::Simulation::Parameters
{
    static constexpr auto restDensity = 1000.F;

    const auto mass = totalMass / static_cast<float>(particleCount);
    const auto particleVolume = mass / restDensity;
    const auto particleSpacing = glm::pow(particleVolume, 1.F / 3.F);
    const auto smoothingRadius = 6.F * particleSpacing;
    const auto particleRadius = smoothingRadius / 8.F;

    return {
        .domain = domain,
        .gravity = glm::vec3 {0.F, 9.81F, 0.F},
        .restDensity = restDensity,
        .pressureConstant = 500.F,
        .nearPressureConstant = 2.F,
        .restitution = 0.8F,
        .smoothingRadius = smoothingRadius,
        .viscosityConstant = .005F,
        .surfaceTensionCoefficient = 0.0F,
        .maxVelocity = 5.F,
        .particleRadius = particleRadius,
        .particleMass = mass,
        .threadsPerBlock = 256,
    };
}

}
