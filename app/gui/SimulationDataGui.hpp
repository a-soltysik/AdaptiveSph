#pragma once
#include <panda/utils/Signals.h>

#include <cuda/Simulation.cuh>

namespace sph
{
class Window;

class SimulationDataGui
{
public:
    explicit SimulationDataGui(const Window& window, const cuda::Simulation::Parameters& simulationData);

    auto getParameters() const -> const cuda::Simulation::Parameters&;

private:
    auto render() -> void;

    panda::utils::signals::BeginGuiRender::ReceiverT _beginGuiReceiver;
    cuda::Simulation::Parameters _simulationData;
    const Window& _window;
    std::function<uint32_t(uint32_t)> _threadsPerBlockSlider;
};
}
