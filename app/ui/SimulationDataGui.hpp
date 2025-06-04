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

    explicit SimulationDataGui();

    void setAverageNeighbourCount(std::pair<float, float> neighbourCount);
    void setDensityDeviation(DensityDeviation densityDeviation);

private:
    void render();

    void displayAverageNeighborCount(std::pair<float, float> averageNeighbors) const;
    static void displayDensityStatistics(const std::vector<float>& densityDeviations,
                                         uint32_t particleCount,
                                         float restDensity);

    panda::utils::signals::BeginGuiRender::ReceiverT _beginGuiReceiver;
    std::pair<float, float> _averageNeighbourCount = {};
    DensityDeviation _densityDeviation {};
};
}
