// BenchmarkVisualizer.cpp
#include "BenchmarkVisualizer.hpp"

#include <panda/Logger.h>
#include <panda/gfx/Camera.h>
#include <panda/gfx/Light.h>
#include <panda/utils/Signals.h>

#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "mesh/InvertedCube.hpp"

namespace sph::benchmark
{

BenchmarkVisualizer::BenchmarkVisualizer(panda::gfx::vulkan::Context& api,
                                         cuda::Simulation::Parameters::TestCase experimentName,
                                         BenchmarkResult::SimulationType simulationType)
    : _api(api),
      _experimentName(experimentName),
      _simulationType(simulationType)
{
    _scene = std::make_unique<panda::gfx::vulkan::Scene>();
}

void BenchmarkVisualizer::initialize(const cuda::Simulation::Parameters& params)
{
    // Setup scene with domain boundaries and lights
    setupScene(params);
    // Setup the camera with a good view of the simulation
    updateCamera();
    panda::log::Info("Visualization initialized for {} ({} simulation)",
                     std::to_underlying(_experimentName),
                     std::to_underlying(_simulationType));
}

void BenchmarkVisualizer::renderFrame(cuda::Simulation& simulation, float deltaTime)
{
    // Update the simulation time
    _simulationTime += deltaTime;
    // Update the particle count in the scene
    _scene->setParticleCount(simulation.getParticlesCount());
    // Render the frame
    _api.makeFrame(deltaTime, *_scene);
}

void BenchmarkVisualizer::setupScene(const cuda::Simulation::Parameters& params)
{
    // Create the textures for visualization
    auto blueTexture = panda::gfx::vulkan::Texture::getDefaultTexture(_api, {0, 0, 1, 0.3f});
    // Create the mesh for the domain boundary
    auto invertedCubeMesh = mesh::inverted_cube::create(_api, "InvertedCube");
    // Set up the domain boundary
    const auto domain = params.domain;
    auto& object = _scene->setDomain("Domain",
                                     {
                                         panda::gfx::vulkan::Surface {blueTexture.get(), invertedCubeMesh.get()}
    });
    object.transform.rotation = {};
    object.transform.translation = domain.getTranslation();
    object.transform.scale = domain.getScale();
    // Register resources with the API
    _api.registerMesh(std::move(invertedCubeMesh));
    _api.registerTexture(std::move(blueTexture));
    // Add lighting
    auto directionalLight = _scene->addLight<panda::gfx::DirectionalLight>("DirectionalLight");
    if (directionalLight.has_value())
    {
        directionalLight.value().get().makeColorLight({1.0f, 0.9f, 0.9f}, 0.1f, 0.8f, 1.0f, 0.8f);
        directionalLight.value().get().direction = {-1.0f, -2.0f, -1.0f};
    }
    // Add a second light for better illumination
    directionalLight = _scene->addLight<panda::gfx::DirectionalLight>("DirectionalLight#2");
    if (directionalLight.has_value())
    {
        directionalLight.value().get().makeColorLight({0.9f, 0.9f, 1.0f}, 0.1f, 0.7f, 1.0f, 0.7f);
        directionalLight.value().get().direction = {1.0f, -1.0f, 1.0f};
    }
    // Add title text to the scene
    std::string title =
        fmt::format("{}-{} simulation", std::to_underlying(_experimentName), std::to_underlying(_simulationType));
    // Note: In a real implementation, you might want to add text rendering here
}

void BenchmarkVisualizer::updateCamera()
{
    // Set up camera with a good view of the simulation
    auto& camera = _scene->getCamera();

    // For perspective projection
    camera.setPerspectiveProjection(panda::gfx::projection::Perspective {.fovY = glm::radians(45.0f),
                                                                         .aspect = _api.getRenderer().getAspectRatio(),
                                                                         .zNear = 0.1f,
                                                                         .zFar = 100.0f});

    // Position camera based on the simulation type
    // For lid-driven cavity, a view from the side is good
    if (_experimentName == cuda::Simulation::Parameters::TestCase::LidDrivenCavity)
    {
        //camera.setViewYXZ(panda::gfx::view::YXZ {
        //    .position = {1.5f,                 0.5f,                -1.5f},
        //    .rotation = {glm::radians(-20.0f), glm::radians(45.0f), 0.0f }
        //});
        auto cameraObject = panda::gfx::vulkan::Transform {};

        cameraObject.translation = {0, 0, -7};
        _scene->getCamera().setViewYXZ(
            panda::gfx::view::YXZ {.position = cameraObject.translation, .rotation = cameraObject.rotation});
    }
    else if (_experimentName == cuda::Simulation::Parameters::TestCase::PoiseuilleFlow)
    {
        // For Poiseuille flow, position camera to see channel cross-section
        auto cameraObject = panda::gfx::vulkan::Transform {};
        cameraObject.translation = {2.5, 0, -3};
        // Rotate to get a good view of the flow profile
        cameraObject.rotation = {glm::radians(0.F), glm::radians(-30.F), 0.0f};
        _scene->getCamera().setViewYXZ(
            panda::gfx::view::YXZ {.position = cameraObject.translation, .rotation = cameraObject.rotation});
    }
    else if (_experimentName == cuda::Simulation::Parameters::TestCase::TaylorGreenVortex)
    {
        auto cameraObject = panda::gfx::vulkan::Transform {};
        cameraObject.translation = {6.28, -3.14, -8};
        // Rotate to get a good view of the flow profile
        cameraObject.rotation = {glm::radians(0.F), glm::radians(-30.F), 0.0f};
        _scene->getCamera().setViewYXZ(
            panda::gfx::view::YXZ {.position = cameraObject.translation, .rotation = cameraObject.rotation});
    }
    else
    {
        // Default camera position for other experiments
        camera.setViewYXZ(panda::gfx::view::YXZ {
            .position = {0.0f, 0.0f, -5.0f},
            .rotation = {0.0f, 0.0f, 0.0f }
        });
    }
}

}  // namespace sph::benchmark
