#include "SimulationDataGui.hpp"

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <imgui.h>
#include <panda/utils/Signals.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cstdint>
#include <cuda/Simulation.cuh>
#include <glm/ext/vector_float4.hpp>
#include <utility>
#include <vector>

#include "Window.hpp"

namespace sph
{

SimulationDataGui::SimulationDataGui(const Window& window, const cuda::Simulation::Parameters& simulationData)
    : _simulationData {simulationData},
      _window {window}
{
    _beginGuiReceiver = panda::utils::signals::beginGuiRender.connect([this](auto data) {
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        render();

        ImGui::Render();

        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), data.commandBuffer, VK_NULL_HANDLE);
    });
}

auto SimulationDataGui::render() -> void
{
    const auto windowSize =
        ImVec2 {static_cast<float>(_window.getSize().x) / 3, static_cast<float>(_window.getSize().y)};
    ImGui::SetNextWindowPos({static_cast<float>(_window.getSize().x) * 2.F / 3.F, 0}, ImGuiCond_Once);
    ImGui::SetNextWindowSize(windowSize, ImGuiCond_Once);
    ImGui::SetNextWindowCollapsed(true, ImGuiCond_Once);

    ImGui::Begin("Simulation Parameters", nullptr);

    auto translation = _simulationData.domain.getTranslation();
    auto scale = _simulationData.domain.getScale();

    ImGui::DragFloat("Smoothing radius", &_simulationData.baseSmoothingRadius, 0.001F, 0.001F, 1.F);
    ImGui::DragFloat("Rest density", &_simulationData.restDensity, 1.F, 1.F, 3000.F);
    ImGui::DragFloat("Pressure constant", &_simulationData.pressureConstant, .001F, .001F, 10.F);
    ImGui::DragFloat("Near Pressure constant", &_simulationData.nearPressureConstant, .001F, .001F, 10.F);
    ImGui::DragFloat("Viscosity constant", &_simulationData.viscosityConstant, 0.001F, 0.001F, 1.F);
    ImGui::DragFloat("Max speed", &_simulationData.maxVelocity, 0.1F, 0.1F, 10.F);
    ImGui::DragFloat3("Translation", &translation[0], 0.1F, -5.F, 5.F);

    _simulationData.domain = cuda::Simulation::Parameters::Domain {}.fromTransform(translation, scale);
    ImGui::End();

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

    displayAverageNeighborCount(_averageNeighbourCount);
    displayDensityStatistics(_densityDeviation.densityDeviations,
                             _densityDeviation.particleCount,
                             _simulationData.restDensity);
}

void SimulationDataGui::displayAverageNeighborCount(float averageNeighbors)
{
    ImGui::Begin("Simulation Debug Info");
    //NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
    ImGui::Text("Average Neighbors: %.2f", static_cast<double>(averageNeighbors));
    ImGui::End();
}

//NOLINTBEGIN(bugprone-easily-swappable-parameters)
void SimulationDataGui::displayDensityStatistics(const std::vector<glm::vec4>& densityDeviations,
                                                 uint32_t particleCount,
                                                 float restDensity)
//NOLINTEND(bugprone-easily-swappable-parameters)
{
    if (particleCount == 0)
    {
        return;
    }

    ImGui::Begin("Density Statistics");

    double minDeviation = DBL_MAX;
    double maxDeviation = -DBL_MAX;
    double avgDeviation = 0.;
    uint32_t underDensityCount = 0;
    uint32_t overDensityCount = 0;

    for (uint32_t i = 0; i < particleCount; i++)
    {
        const auto deviation = static_cast<double>(densityDeviations[i].x);
        minDeviation = std::min(minDeviation, deviation);
        maxDeviation = std::max(maxDeviation, deviation);
        avgDeviation += deviation;

        if (deviation < -0.1)
        {
            underDensityCount++;
        }
        if (deviation > 0.1)
        {
            overDensityCount++;
        }
    }

    avgDeviation /= static_cast<double>(particleCount);

    //NOLINTBEGIN(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
    ImGui::Text("Density Deviation from Rest Density (%.1f):", static_cast<double>(restDensity));
    ImGui::Text("Min: %.2f%% (%.1f)", minDeviation * 100.0, static_cast<double>(restDensity) * (1.0 + minDeviation));
    ImGui::Text("Max: %.2f%% (%.1f)", maxDeviation * 100.0, static_cast<double>(restDensity) * (1.0 + maxDeviation));
    ImGui::Text("Avg: %.2f%% (%.1f)", avgDeviation * 100.0, static_cast<double>(restDensity) * (1.0 + avgDeviation));

    // Display histogram
    ImGui::Text("Deviation Distribution:");
    ImGui::Text("Under Density (<-10%%): %u particles (%.1f%%)",
                underDensityCount,
                100.0 * underDensityCount / particleCount);
    ImGui::Text("Normal Density (±10%%): %u particles (%.1f%%)",
                particleCount - underDensityCount - overDensityCount,
                100.0 * (particleCount - underDensityCount - overDensityCount) / particleCount);
    ImGui::Text("Over Density (>+10%%): %u particles (%.1f%%)",
                overDensityCount,
                100.0 * overDensityCount / particleCount);
    //NOLINTEND(cppcoreguidelines-pro-type-vararg,hicpp-vararg)

    ImGui::End();
}

auto SimulationDataGui::getParameters() const -> const cuda::Simulation::Parameters&
{
    return _simulationData;
}

auto SimulationDataGui::setAverageNeighbourCount(float neighbourCount) -> void
{
    _averageNeighbourCount = neighbourCount;
}

void SimulationDataGui::setDensityDeviation(DensityDeviation densityDeviation)
{
    _densityDeviation = std::move(densityDeviation);
}
}
