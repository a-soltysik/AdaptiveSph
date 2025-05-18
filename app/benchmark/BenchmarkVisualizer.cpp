#include "BenchmarkVisualizer.hpp"

#include <panda/Logger.h>
#include <panda/gfx/Camera.h>
#include <panda/gfx/Light.h>

#include <glm/trigonometric.hpp>
#include <memory>
#include <utility>

#include "benchmark/MetricsCollector.hpp"
#include "cuda/Simulation.cuh"
#include "mesh/InvertedCube.hpp"
#include "panda/gfx/vulkan/Scene.h"
#include "panda/gfx/vulkan/object/Object.h"
#include "panda/gfx/vulkan/object/Surface.h"
#include "panda/gfx/vulkan/object/Texture.h"

namespace sph::benchmark
{

BenchmarkVisualizer::BenchmarkVisualizer(panda::gfx::vulkan::Context& api,
                                         cuda::Simulation::Parameters::TestCase experimentName,
                                         BenchmarkResult::SimulationType simulationType)
    : _api(api),
      _experimentName(experimentName),
      _simulationType(simulationType)
{
}

void BenchmarkVisualizer::initialize(const cuda::Simulation::Parameters& params)
{
    setupScene(params);
    updateCamera();
    panda::log::Info("Visualization initialized for {} ({} simulation)",
                     std::to_underlying(_experimentName),
                     std::to_underlying(_simulationType));
}

void BenchmarkVisualizer::renderFrame(const cuda::Simulation& simulation)
{
    _scene.setParticleCount(simulation.getParticlesCount());
    _api.makeFrame(_scene);
}

void BenchmarkVisualizer::setupScene(const cuda::Simulation::Parameters& params)
{
    auto blueTexture = panda::gfx::vulkan::Texture::getDefaultTexture(_api, {0, 0, 1, 0.3F});
    auto invertedCubeMesh = mesh::inverted_cube::create(_api, "InvertedCube");
    const auto domain = params.domain;
    auto& object = _scene.setDomain("Domain",
                                    {
                                        panda::gfx::vulkan::Surface {blueTexture.get(), invertedCubeMesh.get()}
    });
    object.transform.rotation = {};
    object.transform.translation = domain.getTranslation();
    object.transform.scale = domain.getScale();

    _api.registerMesh(std::move(invertedCubeMesh));
    _api.registerTexture(std::move(blueTexture));

    auto directionalLight = _scene.addLight<panda::gfx::DirectionalLight>("DirectionalLight");
    if (directionalLight.has_value())
    {
        directionalLight.value().get().makeColorLight({1.0F, 0.9F, 0.9F}, 0.1F, 0.8F, 1.0F, 0.8F);
        directionalLight.value().get().direction = {-1.0F, -2.0F, -1.0F};
    }

    directionalLight = _scene.addLight<panda::gfx::DirectionalLight>("DirectionalLight#2");
    if (directionalLight.has_value())
    {
        directionalLight.value().get().makeColorLight({0.9F, 0.9F, 1.0F}, 0.1F, 0.7F, 1.0F, 0.7F);
        directionalLight.value().get().direction = {1.0F, -1.0F, 1.0F};
    }
}

void BenchmarkVisualizer::updateCamera()
{
    auto& camera = _scene.getCamera();
    camera.setPerspectiveProjection(panda::gfx::projection::Perspective {.fovY = glm::radians(45.0F),
                                                                         .aspect = _api.getRenderer().getAspectRatio(),
                                                                         .zNear = 0.1F,
                                                                         .zFar = 100.0F});

    auto cameraObject = panda::gfx::vulkan::Transform {};
    if (_experimentName == cuda::Simulation::Parameters::TestCase::LidDrivenCavity)
    {
        cameraObject.translation = {0, 0, -7};
        camera.setViewYXZ(
            panda::gfx::view::YXZ {.position = cameraObject.translation, .rotation = cameraObject.rotation});
    }
    else if (_experimentName == cuda::Simulation::Parameters::TestCase::PoiseuilleFlow)
    {
        cameraObject.translation = {2.5, 0, -3};
        cameraObject.rotation = {glm::radians(0.F), glm::radians(-30.F), 0.0F};
        camera.setViewYXZ(
            panda::gfx::view::YXZ {.position = cameraObject.translation, .rotation = cameraObject.rotation});
    }
    else if (_experimentName == cuda::Simulation::Parameters::TestCase::TaylorGreenVortex)
    {
        cameraObject.translation = {6.28, -3.14, -8};
        cameraObject.rotation = {glm::radians(0.F), glm::radians(-30.F), 0.0F};
        camera.setViewYXZ(
            panda::gfx::view::YXZ {.position = cameraObject.translation, .rotation = cameraObject.rotation});
    }
    else
    {
        camera.setViewYXZ(panda::gfx::view::YXZ {
            .position = {0.0F, 0.0F, -5.0F},
            .rotation = {0.0F, 0.0F, 0.0F }
        });
    }
}

}
