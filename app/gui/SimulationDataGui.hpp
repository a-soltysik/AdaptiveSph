#pragma once
#include <panda/utils/Signals.h>

#include <cuda/Simulation.cuh>

namespace sph
{
class Window;

class SimulationDataGui
{
public:
    struct DensityDeviation
    {
        std::vector<glm::vec4> densityDeviations;
        uint32_t particleCount;
        float restDensity;
    };

    explicit SimulationDataGui(const Window& window, const cuda::Simulation::Parameters& simulationData);

    auto getParameters() const -> const cuda::Simulation::Parameters&;
    auto setAverageNeighbourCount(uint32_t neighbourCount) -> void;
    void setDensityDeviation(DensityDeviation densityDeviation);

private:
    auto render() -> void;

    static void displayAverageNeighborCount(float averageNeighbors);
    static void displayDensityStatistics(const std::vector<glm::vec4>& densityDeviations,
                                         uint32_t particleCount,
                                         float restDensity);

    panda::utils::signals::BeginGuiRender::ReceiverT _beginGuiReceiver;
    cuda::Simulation::Parameters _simulationData;
    const Window& _window;
    std::function<uint32_t(uint32_t)> _threadsPerBlockSlider;
    uint32_t _averageNeighbourCount = 0;
    DensityDeviation _densityDeviation {};
};
}
