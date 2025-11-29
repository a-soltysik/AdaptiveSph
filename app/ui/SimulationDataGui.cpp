#include "SimulationDataGui.hpp"

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <imgui.h>
#include <panda/utils/Signals.h>
#include <vulkan/vulkan_core.h>

#include <chrono>
#include <cstdint>
#include <utility>

#include "../../cuda/include/cuda/simulation/Simulation.cuh"

namespace sph
{

SimulationDataGui::SimulationDataGui()
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
    displayDensityStatistics(_densityInfo);
    displayDomainControls();
    displayRefinementControls();
}

void SimulationDataGui::displayAverageNeighborCount(float averageNeighbors) const
{
    ImGui::Begin("Simulation Debug Info");
    //NOLINTBEGIN(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
    ImGui::Text("Average Neighbors: %.2f", static_cast<double>(averageNeighbors));
    ImGui::Text("Particle count: %u",
                _densityInfo.underDensityCount + _densityInfo.normalDensityCount + _densityInfo.overDensityCount);
    //NOLINTEND(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
    ImGui::End();
}

void SimulationDataGui::displayDensityStatistics(const cuda::Simulation::DensityInfo& densityInfo)
{
    ImGui::Begin("Density Statistics");

    const auto minDeviation =
        static_cast<double>((densityInfo.minDensity - densityInfo.restDensity) / densityInfo.restDensity);
    const auto maxDeviation =
        static_cast<double>((densityInfo.maxDensity - densityInfo.restDensity) / densityInfo.restDensity);
    const auto avgDeviation =
        static_cast<double>((densityInfo.averageDensity - densityInfo.restDensity) / densityInfo.restDensity);
    const auto particleCount =
        densityInfo.underDensityCount + densityInfo.normalDensityCount + densityInfo.overDensityCount;

    //NOLINTBEGIN(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
    ImGui::Text("Density Deviation from Rest Density (%.1f):", static_cast<double>(densityInfo.restDensity));
    ImGui::Text("Min: %.2f%% (%.1f)", minDeviation * 100.0, static_cast<double>(densityInfo.minDensity));
    ImGui::Text("Max: %.2f%% (%.1f)", maxDeviation * 100.0, static_cast<double>(densityInfo.maxDensity));
    ImGui::Text("Avg: %.2f%% (%.1f)", avgDeviation * 100.0, static_cast<double>(densityInfo.averageDensity));
    ImGui::Text("Deviation Distribution:");
    ImGui::Text("Under Density (<-10%%): %u particles (%.1f%%)",
                densityInfo.underDensityCount,
                100.0 * densityInfo.underDensityCount / particleCount);
    ImGui::Text("Normal Density (±10%%): %u particles (%.1f%%)",
                densityInfo.normalDensityCount,
                100.0 * densityInfo.normalDensityCount / particleCount);
    ImGui::Text("Over Density (>+10%%): %u particles (%.1f%%)",
                densityInfo.overDensityCount,
                100.0 * densityInfo.overDensityCount / particleCount);
    //NOLINTEND(cppcoreguidelines-pro-type-vararg,hicpp-vararg)

    ImGui::End();
}

auto SimulationDataGui::setAverageNeighbourCount(float neighbourCount) -> void
{
    _averageNeighbourCount = neighbourCount;
}

void SimulationDataGui::setDensityInfo(const cuda::Simulation::DensityInfo& densityInfo)
{
    _densityInfo = densityInfo;
}

void SimulationDataGui::setDomain(const cuda::Simulation::Parameters::Domain& domain)
{
    _domain = domain;
}

void SimulationDataGui::onDomainChanged(DomainChangedCallback callback)
{
    _domainChangedCallback = std::move(callback);
}

void SimulationDataGui::onEnableRefinement(EnableRefinementCallback callback)
{
    _enableRefinementCallback = std::move(callback);
}

void SimulationDataGui::displayDomainControls()
{
    ImGui::Begin("Domain Controls");

    auto domainMin = _domain.min;
    auto domainMax = _domain.max;
    auto friction = _domain.friction;

    bool domainChanged = false;

    //NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
    ImGui::Text("Domain Min:");
    domainChanged |= ImGui::SliderFloat("Min X", &domainMin.x, -10.0F, 0.0F);
    domainChanged |= ImGui::SliderFloat("Min Y", &domainMin.y, -10.0F, 0.0F);
    domainChanged |= ImGui::SliderFloat("Min Z", &domainMin.z, -10.0F, 0.0F);

    ImGui::Separator();

    //NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
    ImGui::Text("Domain Max:");
    domainChanged |= ImGui::SliderFloat("Max X", &domainMax.x, 0.0F, 10.0F);
    domainChanged |= ImGui::SliderFloat("Max Y", &domainMax.y, 0.0F, 10.0F);
    domainChanged |= ImGui::SliderFloat("Max Z", &domainMax.z, 0.0F, 10.0F);

    ImGui::Separator();

    if (domainChanged && _domainChangedCallback)
    {
        const auto newDomain =
            cuda::Simulation::Parameters::Domain {.min = domainMin, .max = domainMax, .friction = friction};
        _domain = newDomain;
        _domainChangedCallback(newDomain);
    }

    ImGui::End();
}

void SimulationDataGui::displayRefinementControls()
{
    ImGui::Begin("Refinement Controls");

    if (ImGui::Button("Enable Adaptive Refinement"))
    {
        if (_enableRefinementCallback)
        {
            _enableRefinementCallback();
        }
    }

    ImGui::End();
}

}
