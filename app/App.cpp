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
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cuda/Simulation.cuh>
#include <filesystem>
#include <glm/common.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/ext/vector_uint2.hpp>
#include <glm/ext/vector_uint3.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/trigonometric.hpp>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "benchmark/BenchmarkManager.hpp"
#include "glm/gtx/string_cast.hpp"
#include "gui/SimulationDataGui.hpp"
#include "input_handler/MouseHandler.hpp"
#include "internal/config.hpp"
#include "mesh/InvertedCube.hpp"
#include "mesh/UvSphere.hpp"
#include "movement_handler/MovementHandler.hpp"
#include "movement_handler/RotationHandler.hpp"
#include "utils/FrameTimeManager.hpp"

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

}

namespace sph
{

App::App(std::string configPath)
    : _configPath(std::move(configPath))
{
}

auto App::run() -> int
{
    initializeLogger();
    registerSignalHandlers();

    loadConfigurationFromFile(_configPath);

    static constexpr auto defaultWidth = uint32_t {1920};
    static constexpr auto defaultHeight = uint32_t {1080};
    _window = std::make_unique<Window>(glm::uvec2 {defaultWidth, defaultHeight}, config::project_name.data());
    _api = std::make_unique<panda::gfx::vulkan::Context>(*_window);

    auto redTexture = panda::gfx::vulkan::Texture::getDefaultTexture(*_api, {1, 0, 0, 1});

    _scene = std::make_unique<panda::gfx::vulkan::Scene>();

    _api->registerTexture(std::move(redTexture));

    setDefaultScene();

    if (_benchmarkParams.has_value() && _benchmarkParams.value().enabled)
    {
        runBenchmarks();
        return 0;
    }

    _simulation = createSimulation(_simulationParameters,
                                   _particles,
                                   _api->initializeParticleSystem(_refinementParameters.maxParticleCount),
                                   _refinementParameters);

    mainLoop();
    return 0;
}

auto App::loadConfigurationFromFile(const std::string& configPath) -> bool
{
    if (!std::filesystem::exists(configPath))
    {
        panda::log::Warning("Config file not found: {}. Using default parameters.", configPath);
        return false;
    }

    if (!_configManager.loadFromFile(configPath))
    {
        panda::log::Warning("Failed to load configuration from file: {}. Using default parameters.", configPath);
        return false;
    }

    panda::log::Info("Successfully loaded configuration from file: {}", configPath);

    //NOLINTBEGIN(bugprone-unchecked-optional-access)
    _simulationParameters = _configManager.getSimulationParameters().value();
    _initialParameters = _configManager.getInitialParameters().value();
    _refinementParameters = _configManager.getRefinementParameters().value();
    _benchmarkParams = _configManager.getBenchmarkParameters();
    //NOLINTEND(bugprone-unchecked-optional-access)

    return true;
}

auto App::runBenchmarks() -> void
{
    if (!_benchmarkParams.has_value() || !_benchmarkParams.value().enabled)
    {
        panda::log::Info("Benchmarks are disabled.");
        return;
    }

    panda::log::Info("Running benchmarks...");
    // Ensure window is created and visible
    if (!_window)
    {
        _window = std::make_unique<Window>(glm::uvec2 {1280, 720}, "SPH Benchmark");
    }

    // Create API if not already created
    if (!_api)
    {
        _api = std::make_unique<panda::gfx::vulkan::Context>(*_window);
    }

    // Create and run the benchmark manager
    benchmark::BenchmarkManager benchmarkManager;
    benchmarkManager.runBenchmarks(_benchmarkParams.value(), _simulationParameters, *_api, *_window);

    panda::log::Info("Benchmarks completed.");
}

auto App::mainLoop() const -> void
{
    auto cameraObject = panda::gfx::vulkan::Transform {};

    cameraObject.translation = {0, 0.5, -5};
    _scene->getCamera().setViewYXZ(
        panda::gfx::view::YXZ {.position = cameraObject.translation, .rotation = cameraObject.rotation});

    auto gui = SimulationDataGui {*_window};

    auto timeManager = FrameTimeManager {};
    const std::vector densityDeviations(_simulation->getParticlesCount(), 0.F);
    while (!_window->shouldClose()) [[likely]]
    {
        if (!_window->isMinimized()) [[likely]]
        {
            panda::utils::signals::gameLoopIterationStarted.registerSender()();
            _window->processInput();

            timeManager.update();
            _simulation->update(timeManager.getDelta());
            gui.setAverageNeighbourCount(_simulation->calculateAverageNeighborCount());
            gui.setDensityDeviation({.densityDeviations = _simulation->updateDensityDeviations(),
                                     .particleCount = _simulation->getParticlesCount(),
                                     .restDensity = _simulationParameters.restDensity});

            _scene->setParticleCount(_simulation->getParticlesCount());
            _scene->getCamera().setPerspectiveProjection(
                panda::gfx::projection::Perspective {.fovY = glm::radians(50.F),
                                                     .aspect = _api->getRenderer().getAspectRatio(),
                                                     .zNear = 0.1F,
                                                     .zFar = 100});
            processCamera(timeManager.getDelta(), *_window, cameraObject, _scene->getCamera());

            _api->makeFrame(timeManager.getDelta(), *_scene);
        }
        else [[unlikely]]
        {
            _window->waitForInput();
        }
    }

    panda::log::Info("Main loop ended. Average frame rate: {} FPS", timeManager.getMeanFrameRate());
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

    createDomainBoundaries();
    createParticleDistribution();
    setupLighting();

    _api->registerMesh(std::move(sphereMesh));
    _api->registerMesh(std::move(invertedCubeMesh));
    _api->registerTexture(std::move(redTexture));
    _api->registerTexture(std::move(blueTexture));
}

void App::createDomainBoundaries()
{
    auto blueTexture = panda::gfx::vulkan::Texture::getDefaultTexture(*_api, {0, 0, 1, 0.3f});
    auto invertedCubeMesh = mesh::inverted_cube::create(*_api, "InvertedCube");

    const auto domain = _simulationParameters.domain;
    auto& object = _scene->setDomain("Domain",
                                     {
                                         panda::gfx::vulkan::Surface {blueTexture.get(), invertedCubeMesh.get()}
    });
    object.transform.rotation = {};
    object.transform.translation = domain.getTranslation();
    object.transform.scale = domain.getScale();

    _api->registerMesh(std::move(invertedCubeMesh));
    _api->registerTexture(std::move(blueTexture));
}

void App::createParticleDistribution()
{
    const glm::uvec3 simulationSize = _initialParameters.particleCount;

    panda::log::Info("Creating particle distribution with grid size: {}x{}x{}",
                     simulationSize.x,
                     simulationSize.y,
                     simulationSize.z);

    const glm::vec3 domainMin = _simulationParameters.domain.min;
    const glm::vec3 domainMax = _simulationParameters.domain.max;
    const glm::vec3 domainSize = domainMax - domainMin;
    const glm::uvec3 gridSize = _initialParameters.particleCount;

    glm::vec3 particleSpacing = calculateParticleSpacing(domainSize, gridSize);
    panda::log::Info("Particle spacing: {}", glm::to_string(particleSpacing));

    const glm::vec3 startPos = domainMin + particleSpacing;
    createParticlesInGrid(startPos, gridSize, particleSpacing);
}

glm::vec3 App::calculateParticleSpacing(const glm::vec3& domainSize, const glm::uvec3& gridSize)
{
    glm::vec3 particleSpacing;
    particleSpacing.x = gridSize.x > 1 ? domainSize.x / static_cast<float>(gridSize.x + 1) : domainSize.x / 2.0F;
    particleSpacing.y = gridSize.y > 1 ? domainSize.y / static_cast<float>(gridSize.y + 1) : domainSize.y / 2.0F;
    particleSpacing.z = gridSize.z > 1 ? domainSize.z / static_cast<float>(gridSize.z + 1) : domainSize.z / 2.0F;

    return particleSpacing;
}

void App::createParticlesInGrid(const glm::vec3& startPos, const glm::uvec3& gridSize, const glm::vec3& spacing)
{
    _particles.clear();
    _particles.reserve(gridSize.x * gridSize.y * gridSize.z);
    for (uint32_t i = 0; i < gridSize.x; i++)
    {
        for (uint32_t j = 0; j < gridSize.y; j++)
        {
            for (uint32_t k = 0; k < gridSize.z; k++)
            {
                const auto position = startPos + glm::vec3(spacing.x * static_cast<float>(i),
                                                           spacing.y * static_cast<float>(j),
                                                           spacing.z * static_cast<float>(k));

                _particles.emplace_back(position, 0.F);
            }
        }
    }
}

void App::setupLighting() const
{
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

}
