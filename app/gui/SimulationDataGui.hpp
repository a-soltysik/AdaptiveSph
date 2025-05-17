#pragma once
#include <panda/utils/Signals.h>

#include <cstdint>
#include <cuda/Simulation.cuh>
#include <glm/ext/vector_float4.hpp>
#include <vector>

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

    explicit SimulationDataGui(const Window& window);

    auto setAverageNeighbourCount(float neighbourCount) -> void;
    void setDensityDeviation(DensityDeviation densityDeviation);

private:
    auto render() -> void;

    static void displayAverageNeighborCount(float averageNeighbors);
    static void displayDensityStatistics(const std::vector<glm::vec4>& densityDeviations,
                                         uint32_t particleCount,
                                         float restDensity);

    panda::utils::signals::BeginGuiRender::ReceiverT _beginGuiReceiver;
    const Window& _window;
    float _averageNeighbourCount = 0.F;
    DensityDeviation _densityDeviation {};
};
}
