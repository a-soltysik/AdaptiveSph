#include "App.hpp"

#include <GLFW/glfw3.h>
#include <fmt/format.h>
#include <panda/Logger.h>
#include <panda/gfx/Camera.h>
#include <panda/gfx/Light.h>
#include <panda/gfx/object/Object.h>
#include <panda/gfx/object/Surface.h>
#include <panda/gfx/object/Texture.h>
#include <panda/utils/Assert.h>
#include <panda/utils/Signals.h>

#include <array>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cuda/simulation/Simulation.cuh>
#include <cuda/simulation/physics/StaticBoundaryDomain.cuh>
#include <glm/common.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_uint2.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/trigonometric.hpp>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "input/MouseHandler.hpp"
#include "internal/config.hpp"
#include "mesh/InvertedCube.hpp"
#include "movement/MovementHandler.hpp"
#include "movement/RotationHandler.hpp"
#include "ui/SimulationDataGui.hpp"
#include "utils/Configuration.hpp"
#include "utils/FrameTimeManager.hpp"
#include "utils/InitialParticleGrid.hpp"
#include "utils/TaskScheduler.hpp"

namespace
{
struct AdaptiveRefinementStatistics
{
    uint32_t passCount {};
    uint32_t mergedPairCount {};
    uint32_t splitParentCount {};
    uint32_t particlesAfterLastPass {};
};

constexpr auto simulationTimeStep = 0.0001F;

auto generateBoundaryDomain(const sph::cuda::Simulation::Parameters& simulationParameters,
                            const sph::cuda::ScenarioParameters& scenarioParameters,
                            const sph::cuda::Simulation::Parameters::Domain& domain)
    -> sph::cuda::physics::StaticBoundaryDomain
{
    const auto particleSpacing = simulationParameters.baseParticleRadius * 2;
    auto wallFlags = sph::cuda::physics::StaticBoundaryDomain::WallFlags::All;
    const auto periodicX = scenarioParameters.type == sph::cuda::ScenarioType::PoiseuilleFlow;

    if (periodicX)
    {
        // Poiseuille flow wraps X, so its boundary particles only cover the Y and Z walls.
        wallFlags = sph::cuda::physics::StaticBoundaryDomain::WallFlags::MinY |
                    sph::cuda::physics::StaticBoundaryDomain::WallFlags::MaxY |
                    sph::cuda::physics::StaticBoundaryDomain::WallFlags::MinZ |
                    sph::cuda::physics::StaticBoundaryDomain::WallFlags::MaxZ;
    }

    return sph::cuda::physics::StaticBoundaryDomain::generate(domain,
                                                              particleSpacing,
                                                              simulationParameters.restDensity,
                                                              simulationParameters.baseSmoothingRadius,
                                                              wallFlags,
                                                              periodicX);
}

auto recordAdaptiveRefinementPass(AdaptiveRefinementStatistics& statistics,
                                  const std::optional<sph::cuda::Simulation::AdaptiveRefinementPass>& refinementPass)
    -> void
{
    if (!refinementPass.has_value())
    {
        return;
    }

    ++statistics.passCount;
    statistics.mergedPairCount += refinementPass->mergedPairs;
    statistics.splitParentCount += refinementPass->splitParents;
    statistics.particlesAfterLastPass = refinementPass->particlesAfter;

    if (refinementPass->particlesBefore == refinementPass->particlesAfter)
    {
        return;
    }

    panda::log::Info(
        "Adaptive refinement changed fluid particle count from {} to {} ({} merge pairs, {} split parents)",
        refinementPass->particlesBefore,
        refinementPass->particlesAfter,
        refinementPass->mergedPairs,
        refinementPass->splitParents);
}

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
                   panda::gfx::Transform& cameraObject,
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

    auto config = loadConfigurationFromFile(_configPath);

    const auto initialParameters = panda::expect(config.initialParameters, "Initial parameters couldn't be found");
    _simulationParameters = panda::expect(config.simulationParameters, "Simulation parameters couldn't be found");
    _scenarioParameters = config.scenarioParameters.value_or(cuda::ScenarioParameters {});
    _refinementParameters = config.refinementParameters;

    static constexpr auto defaultWidth = uint32_t {1920};
    static constexpr auto defaultHeight = uint32_t {1080};
    _window =
        std::make_unique<Window>(glm::uvec2 {defaultWidth, defaultHeight}, std::string {config::project_name}.data());
    _api = std::make_unique<panda::gfx::Context>(*_window);

    _scene = std::make_unique<panda::gfx::Scene>();

    setDefaultScene(_simulationParameters, initialParameters);

    if (_scenarioParameters.type == cuda::ScenarioType::PoiseuilleFlow)
    {
        panda::log::Info("Poiseuille flow scenario: using periodic boundaries in X direction");
    }

    const auto boundaryDomain =
        generateBoundaryDomain(_simulationParameters, _scenarioParameters, _simulationParameters.domain);

    panda::log::Info("Generated {} boundary particles for static domain", boundaryDomain.getParticleCount());

    constexpr auto boundaryCapacityMultiplier = uint32_t {10};
    panda::expect(
        boundaryDomain.getParticleCount() <= std::numeric_limits<uint32_t>::max() / boundaryCapacityMultiplier,
        "Boundary particle capacity exceeds the supported range");
    const auto maxBoundaryCapacity = boundaryDomain.getParticleCount() * boundaryCapacityMultiplier;

    _simulation = cuda::createSimulation(
        _simulationParameters,
        _particles,
        _api->initializeParticleSystem(config.refinementParameters
                                           .and_then([](const auto& refinementParameter) -> std::optional<uint32_t> {
                                               if (refinementParameter.enabled)
                                               {
                                                   return refinementParameter.maxParticleCount;
                                               }
                                               return std::nullopt;
                                           })
                                           .value_or(initialParameters.getScalarCount())),
        _api->initializeBoundaryParticleSystem(
            maxBoundaryCapacity,
            config.renderingParameters.value_or(utils::RenderingParameters {}).renderBoundaryParticles),
        boundaryDomain,
        config.refinementParameters,
        initialParameters.getScalarCount(),
        maxBoundaryCapacity,
        _scenarioParameters);

    mainLoop();
    return 0;
}

auto App::loadConfigurationFromFile(const std::string& configPath) -> utils::Configuration
{
    auto config = panda::expect(utils::loadConfigurationFromFile(configPath),
                                fmt::format("Could not load configuration file: {}", configPath));
    utils::validateConfiguration(config);

    panda::log::Info("Successfully loaded configuration from file: {}", configPath);

    return config;
}

auto App::configureSimulationGui(SimulationDataGui& gui) -> void
{
    gui.setDomain(_simulationParameters.domain);
    gui.onDomainChanged([this](const cuda::Simulation::Parameters::Domain& newDomain) {
        updateDomain(newDomain);
    });
    if (_refinementParameters.has_value() && _refinementParameters->enabled)
    {
        gui.onRequestAdaptiveRefinement([this] {
            _simulation->requestAdaptiveRefinement();
        });
    }
}

auto App::mainLoop() -> void
{
    auto cameraObject = panda::gfx::Transform {};

    cameraObject.translation = {0, 0.5, -5};
    _scene->getCamera().setViewYXZ(
        panda::gfx::view::YXZ {.position = cameraObject.translation, .rotation = cameraObject.rotation});

    auto gui = SimulationDataGui {};
    auto refinementStatistics = AdaptiveRefinementStatistics {};
    configureSimulationGui(gui);
    auto timeManager = utils::FrameTimeManager {};
    auto taskScheduler = utils::TaskScheduler {};
    taskScheduler.addTask(
        [&gui, this] {
            gui.setAverageNeighbourCount(_simulation->calculateAverageNeighborCount());
        },
        std::chrono::seconds {1});
    taskScheduler.addTask(
        [&gui, this] {
            gui.setDensityInfo(_simulation->getDensityInfo(0.1F));
        },
        std::chrono::seconds {1});
    taskScheduler.addTask(
        [&] {
            if (refinementStatistics.passCount == 0)
            {
                return;
            }

            panda::log::Info("Adaptive refinement ran {} passes: {} merge pairs, {} split parents; fluid particles: {}",
                             refinementStatistics.passCount,
                             refinementStatistics.mergedPairCount,
                             refinementStatistics.splitParentCount,
                             refinementStatistics.particlesAfterLastPass);
            refinementStatistics.passCount = 0;
            refinementStatistics.mergedPairCount = 0;
            refinementStatistics.splitParentCount = 0;
        },
        std::chrono::seconds {1});

    while (!_window->shouldClose()) [[likely]]
    {
        if (!_window->isMinimized()) [[likely]]
        {
            panda::utils::signals::gameLoopIterationStarted.registerSender()();
            _window->processInput();

            timeManager.update();

            _api->waitForSimulationAccess();

            updateScenario();
            _simulation->update(simulationTimeStep);

            recordAdaptiveRefinementPass(refinementStatistics, _simulation->takeAdaptiveRefinementPass());

            taskScheduler.update(std::chrono::milliseconds {static_cast<uint32_t>(timeManager.getDelta() * 1000)});
            _scene->setParticleCount(_simulation->getFluidParticlesCount());
            _scene->setBoundaryParticleCount(_simulation->getBoundaryParticlesCount());

            _scene->getCamera().setPerspectiveProjection(
                panda::gfx::projection::Perspective {.fovY = glm::radians(50.F),
                                                     .aspect = _api->getAspectRatio(),
                                                     .zNear = 0.1F,
                                                     .zFar = 100});
            processCamera(timeManager.getDelta(), *_window, cameraObject, _scene->getCamera());

            _api->renderFrame(*_scene);
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

void App::setDefaultScene(const cuda::Simulation::Parameters& simulationParameters,
                          const utils::InitialParameters& initialParameters)
{
    createDomainBoundaries(simulationParameters.domain);
    createParticleDistribution(initialParameters);
    setupLighting();
}

void App::createDomainBoundaries(const cuda::Simulation::Parameters::Domain& domain) const
{
    auto blueTexture = panda::gfx::Texture::getDefaultTexture(*_api, {0.25, 0.25, 0.3, 1.F});
    auto invertedCubeMesh = mesh::inverted_cube::create(*_api, "InvertedCube");

    auto& object = _scene->setDomain("Domain",
                                     {
                                         panda::gfx::Surface {blueTexture.get(), invertedCubeMesh.get()}
    });
    object.transform.rotation = {};
    object.transform.translation = domain.getTranslation();
    object.transform.scale = domain.getScale();

    _api->registerMesh(std::move(invertedCubeMesh));
    _api->registerTexture(std::move(blueTexture));
}

void App::createParticleDistribution(const utils::InitialParameters& initialParameters)
{
    panda::log::Info("Creating particle distribution with grid size: {}",
                     glm::to_string(initialParameters.particleCount));

    createParticlesInGrid(initialParameters);
}

void App::createParticlesInGrid(const utils::InitialParameters& initialParameters)
{
    _particles.clear();
    _particles.reserve(initialParameters.getScalarCount());

    const auto xRange = utils::getInitialParticleGridRange(_simulationParameters.domain.min.x,
                                                           _simulationParameters.domain.max.x,
                                                           1.F,
                                                           _simulationParameters.baseParticleRadius,
                                                           false);
    const auto yRange = utils::getInitialParticleGridRange(_simulationParameters.domain.min.y,
                                                           _simulationParameters.domain.max.y,
                                                           initialParameters.verticalFillRatio,
                                                           _simulationParameters.baseParticleRadius,
                                                           true);
    const auto zRange = utils::getInitialParticleGridRange(_simulationParameters.domain.min.z,
                                                           _simulationParameters.domain.max.z,
                                                           1.F,
                                                           _simulationParameters.baseParticleRadius,
                                                           false);

    for (uint32_t i = 0; i < initialParameters.particleCount.x; i++)
    {
        for (uint32_t j = 0; j < initialParameters.particleCount.y; j++)
        {
            for (uint32_t k = 0; k < initialParameters.particleCount.z; k++)
            {
                const auto position =
                    glm::vec3 {utils::getInitialParticleGridCoordinate(xRange, initialParameters.particleCount.x, i),
                               utils::getInitialParticleGridCoordinate(yRange, initialParameters.particleCount.y, j),
                               utils::getInitialParticleGridCoordinate(zRange, initialParameters.particleCount.z, k)};

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

void App::updateDomain(const cuda::Simulation::Parameters::Domain& newDomain)
{
    panda::log::Info("Updating domain: min={}, max={}", glm::to_string(newDomain.min), glm::to_string(newDomain.max));

    _simulationParameters.domain = newDomain;

    const auto boundaryDomain = generateBoundaryDomain(_simulationParameters, _scenarioParameters, newDomain);

    panda::log::Info("Generated {} boundary particles for new domain", boundaryDomain.getParticleCount());

    _simulation->updateDomain(newDomain, boundaryDomain);

    auto& domainObject = _scene->getDomain();
    domainObject.transform.translation = newDomain.getTranslation();
    domainObject.transform.scale = newDomain.getScale();
}

void App::updateScenario()
{
    if (_scenarioParameters.type == cuda::ScenarioType::Unspecified)
    {
        return;
    }
    const float currentTime = _simulation->getCurrentSimulationTime();
    if (!_scenarioActivated && currentTime >= _scenarioParameters.cooldown)
    {
        _scenarioActivated = true;
        _simulation->setScenarioActivated(true);
        panda::log::Info("Scenario activated after cooldown of {:.2f}s", _scenarioParameters.cooldown);

        if (_scenarioParameters.type == cuda::ScenarioType::DamBreak)
        {
            if (_scenarioParameters.damBreak.has_value())
            {
                updateDomain(_scenarioParameters.damBreak->expandedDomain);
            }
        }
    }
}
}
