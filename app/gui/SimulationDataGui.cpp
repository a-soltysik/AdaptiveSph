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

SimulationDataGui::SimulationDataGui(const Window& window)
    : _window {window}
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
                             _densityDeviation.restDensity);
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
    double avgDeviation = 0.0;
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

auto SimulationDataGui::setAverageNeighbourCount(float neighbourCount) -> void
{
    _averageNeighbourCount = neighbourCount;
}

void SimulationDataGui::setDensityDeviation(DensityDeviation densityDeviation)
{
    _densityDeviation = std::move(densityDeviation);
}
}
