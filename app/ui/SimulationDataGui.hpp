#pragma once
#include <panda/utils/Signals.h>

#include <../../cuda/include/cuda/simulation/Simulation.cuh>
#include <functional>
#include <glm/ext/vector_float3.hpp>

namespace sph
{
class Window;

class SimulationDataGui
{
public:
    using DomainChangedCallback = std::function<void(const cuda::Simulation::Parameters::Domain&)>;
    using EnableRefinementCallback = std::function<void()>;

    explicit SimulationDataGui();

    void setAverageNeighbourCount(float neighbourCount);
    void setDensityInfo(const cuda::Simulation::DensityInfo& densityInfo);
    void setDomain(const cuda::Simulation::Parameters::Domain& domain);

    void onDomainChanged(DomainChangedCallback callback);
    void onEnableRefinement(EnableRefinementCallback callback);

private:
    void render();

    void displayAverageNeighborCount(float averageNeighbors) const;
    static void displayDensityStatistics(const cuda::Simulation::DensityInfo& densityInfo);
    void displayDomainControls();
    void displayRefinementControls();

    panda::utils::signals::BeginGuiRender::ReceiverT _beginGuiReceiver;
    float _averageNeighbourCount = 0.F;
    cuda::Simulation::DensityInfo _densityInfo {};
    cuda::Simulation::Parameters::Domain _domain {};
    DomainChangedCallback _domainChangedCallback;
    EnableRefinementCallback _enableRefinementCallback;
};
}
