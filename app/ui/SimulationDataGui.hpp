#pragma once
#include <panda/utils/Signals.h>

#include <cuda/Simulation.cuh>

namespace sph
{
class Window;

class SimulationDataGui
{
public:
    explicit SimulationDataGui();

    void setAverageNeighbourCount(float neighbourCount);
    void setDensityInfo(const cuda::Simulation::DensityInfo& densityInfo);

private:
    void render() const;

    void displayAverageNeighborCount(float averageNeighbors) const;
    static void displayDensityStatistics(const cuda::Simulation::DensityInfo& densityInfo);

    panda::utils::signals::BeginGuiRender::ReceiverT _beginGuiReceiver;
    float _averageNeighbourCount = 0.F;
    cuda::Simulation::DensityInfo _densityInfo {};
};
}
