#pragma once
#include <panda/utils/Signals.h>

#include <vector>

namespace sph
{
class Window;

class SimulationDataGui
{
public:
    struct DensityDeviation
    {
        std::vector<float> densityDeviations;
        uint32_t particleCount;
        float restDensity;
    };

    explicit SimulationDataGui(const Window& window);

    void setAverageNeighbourCount(float neighbourCount);
    void setDensityDeviation(DensityDeviation densityDeviation);

private:
    void render();

    void displayAverageNeighborCount(float averageNeighbors) const;
    static void displayDensityStatistics(const std::vector<float>& densityDeviations,
                                         uint32_t particleCount,
                                         float restDensity);

    panda::utils::signals::BeginGuiRender::ReceiverT _beginGuiReceiver;
    const Window& _window;
    float _averageNeighbourCount = 0.F;
    DensityDeviation _densityDeviation {};
};
}
