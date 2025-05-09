#include "SimulationDataGui.hpp"

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <imgui.h>
#include <panda/Logger.h>
#include <panda/utils/Signals.h>
#include <vulkan/vulkan_core.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cuda/Simulation.cuh>
#include <glm/common.hpp>
#include <glm/exponential.hpp>

#include "Window.hpp"

namespace sph
{

SimulationDataGui::SimulationDataGui(const Window& window, const cuda::Simulation::Parameters& simulationData)
    : _simulationData {simulationData},
      _window {window},
      _threadsPerBlockSlider {[rawThreads = static_cast<int>(
                                   glm::round(glm::log2(static_cast<float>(_simulationData.threadsPerBlock) / 32.F)))](
                                  uint32_t threadsPerBlock) mutable {
          static constexpr auto powers = std::array {"32", "64", "128", "256", "512", "1024"};

          rawThreads = static_cast<int>(glm::round(glm::log2(static_cast<float>(threadsPerBlock) / 32.F)));
          ImGui::SliderInt("Threads per block", &rawThreads, 0, 5, powers[static_cast<uint32_t>(rawThreads)]);
          return 32U << static_cast<uint32_t>(rawThreads);
      }}
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
    const auto newThreadsPerBlock = _threadsPerBlockSlider(_simulationData.threadsPerBlock);

    if (newThreadsPerBlock != _simulationData.threadsPerBlock)
    {
        panda::log::Info("Number of threads per blocked has changed from {} to {}",
                         _simulationData.threadsPerBlock,
                         newThreadsPerBlock);
    }
    _simulationData.threadsPerBlock = newThreadsPerBlock;

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
    ImGui::Text("Average Neighbors: %.2f", averageNeighbors);

    // Add color coding for quick visual feedback
    ImGui::SameLine();
    if (averageNeighbors < 15.0f)
    {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "(Low)");
    }
    else if (averageNeighbors > 60.0f)
    {
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "(High)");
    }
    else
    {
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "(Good)");
    }

    ImGui::End();
}

void SimulationDataGui::displayDensityStatistics(const std::vector<glm::vec4>& densityDeviations,
                                                 uint32_t particleCount,
                                                 float restDensity)
{
    if (particleCount == 0)
    {
        return;
    }

    ImGui::Begin("Density Statistics");

    // Calculate min, max, and average deviation
    float minDeviation = FLT_MAX;
    float maxDeviation = -FLT_MAX;
    float avgDeviation = 0.0f;
    int underDensityCount = 0;
    int overDensityCount = 0;

    for (uint32_t i = 0; i < particleCount; i++)
    {
        float deviation = densityDeviations[i].x;
        minDeviation = std::min(minDeviation, deviation);
        maxDeviation = std::max(maxDeviation, deviation);
        avgDeviation += deviation;

        if (deviation < -0.1f)
        {
            underDensityCount++;
        }
        if (deviation > 0.1f)
        {
            overDensityCount++;
        }
    }

    avgDeviation /= static_cast<float>(particleCount);

    // Display statistics
    ImGui::Text("Density Deviation from Rest Density (%.1f):", restDensity);
    ImGui::Text("Min: %.2f%% (%.1f)", minDeviation * 100.0f, restDensity * (1.0f + minDeviation));
    ImGui::Text("Max: %.2f%% (%.1f)", maxDeviation * 100.0f, restDensity * (1.0f + maxDeviation));
    ImGui::Text("Avg: %.2f%% (%.1f)", avgDeviation * 100.0f, restDensity * (1.0f + avgDeviation));

    // Display histogram
    ImGui::Text("Deviation Distribution:");
    ImGui::Text("Under Density (<-10%%): %d particles (%.1f%%)",
                underDensityCount,
                100.0f * underDensityCount / particleCount);
    ImGui::Text("Normal Density (±10%%): %d particles (%.1f%%)",
                particleCount - underDensityCount - overDensityCount,
                100.0f * (particleCount - underDensityCount - overDensityCount) / particleCount);
    ImGui::Text("Over Density (>+10%%): %d particles (%.1f%%)",
                overDensityCount,
                100.0f * overDensityCount / particleCount);

    // Create a color bar to visualize the range
    const float barWidth = ImGui::GetContentRegionAvail().x;
    const float barHeight = 20.0f;
    const ImVec2 barPos = ImGui::GetCursorScreenPos();

    ImDrawList* drawList = ImGui::GetWindowDrawList();

    // Draw gradient bar
    for (int i = 0; i < barWidth; i++)
    {
        float t = static_cast<float>(i) / barWidth;
        float deviation = minDeviation + t * (maxDeviation - minDeviation);

        ImColor color;
        if (deviation < -0.2f)
        {
            color = ImColor(0.0f, 0.0f, 1.0f);  // Blue for very low density
        }
        else if (deviation < -0.1f)
        {
            color = ImColor(0.0f, 0.5f + (deviation + 0.2f) * 5.0f, 1.0f);  // Cyan for low density
        }
        else if (deviation < 0.1f)
        {
            color = ImColor(0.0f, 1.0f, 0.0f);  // Green for normal density
        }
        else if (deviation < 0.2f)
        {
            color = ImColor(1.0f, 1.0f - (deviation - 0.1f) * 10.0f, 0.0f);  // Yellow to orange for high density
        }
        else
        {
            color = ImColor(1.0f, 0.0f, 0.0f);  // Red for very high density
        }

        drawList->AddRectFilled(ImVec2(barPos.x + i, barPos.y), ImVec2(barPos.x + i + 1, barPos.y + barHeight), color);
    }

    // Add labels to the color bar
    drawList->AddText(ImVec2(barPos.x, barPos.y + barHeight + 5),
                      ImColor(1.0f, 1.0f, 1.0f),
                      (std::to_string(static_cast<int>(minDeviation * 100)) + "%").c_str());
    drawList->AddText(ImVec2(barPos.x + barWidth - 40, barPos.y + barHeight + 5),
                      ImColor(1.0f, 1.0f, 1.0f),
                      (std::to_string(static_cast<int>(maxDeviation * 100)) + "%").c_str());
    drawList->AddText(ImVec2(barPos.x + barWidth / 2 - 20, barPos.y + barHeight + 5),
                      ImColor(1.0f, 1.0f, 1.0f),
                      (std::to_string(static_cast<int>(avgDeviation * 100)) + "%").c_str());

    ImGui::Dummy(ImVec2(0, barHeight + 20));  // Add space after the bar

    // Particle coloring legend
    ImGui::Separator();
    ImGui::Text("Particle Color Legend:");

    const float legendColorSize = 20.0f;
    auto drawColorLabel = [&](const char* label, ImColor color) {
        ImVec2 p = ImGui::GetCursorScreenPos();
        drawList->AddRectFilled(p, ImVec2(p.x + legendColorSize, p.y + legendColorSize), color);
        ImGui::Dummy(ImVec2(legendColorSize, legendColorSize));
        ImGui::SameLine();
        ImGui::Text("%s", label);
    };

    drawColorLabel("Very Low Density (<-20%)", ImColor(0.0f, 0.0f, 1.0f));
    drawColorLabel("Low Density (-20% to -10%)", ImColor(0.0f, 0.75f, 1.0f));
    drawColorLabel("Normal Density (±10%)", ImColor(0.0f, 1.0f, 0.0f));
    drawColorLabel("High Density (10% to 20%)", ImColor(1.0f, 0.5f, 0.0f));
    drawColorLabel("Very High Density (>20%)", ImColor(1.0f, 0.0f, 0.0f));

    ImGui::End();
}

auto SimulationDataGui::getParameters() const -> const cuda::Simulation::Parameters&
{
    return _simulationData;
}

auto SimulationDataGui::setAverageNeighbourCount(uint32_t neighbourCount) -> void
{
    _averageNeighbourCount = neighbourCount;
}

void SimulationDataGui::setDensityDeviation(DensityDeviation densityDeviation)
{
    _densityDeviation = densityDeviation;
}
}
