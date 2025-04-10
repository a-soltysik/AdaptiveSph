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

    ImGui::DragFloat("Rest density", &_simulationData.restDensity, 1.F, 1.F, 1000.F);
    ImGui::DragFloat("Pressure constant", &_simulationData.pressureConstant, 1.F, 1.F, 1000.F);
    ImGui::DragFloat("Near Pressure constant", &_simulationData.nearPressureConstant, 1.F, 1.F, 1000.F);
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
}

auto SimulationDataGui::getParameters() const -> const cuda::Simulation::Parameters&
{
    return _simulationData;
}
}
