#pragma once

#include <cuda/simulation/Simulation.cuh>
#include <string>

#include "ExperimentData.hpp"
#include "utils/Configuration.hpp"

namespace sph::experiment
{

class ExperimentDataCollector
{
public:
    ExperimentDataCollector(cuda::Simulation* simulation,
                            const std::string& configPath,
                            const utils::ExperimentConfig& config,
                            const std::string& scenarioType,
                            bool refinementEnabled,
                            const std::string& refinementCriterion,
                            float cooldownTime);

    void onFrameCompleted(float computationTimeMs);
    void finalize();

    [[nodiscard]] auto isEnabled() const -> bool
    {
        return _config.enabled;
    }

private:
    void collectPerFrameMetrics(float computationTimeMs);
    void collectFinalState();
    void calculateStatistics();
    void saveToJson();

    auto generateOutputPath() const -> std::string;
    auto generateExperimentName() const -> std::string;
    auto getCurrentTimestamp() const -> std::string;

    cuda::Simulation* _simulation;
    ExperimentData _data;
    utils::ExperimentConfig _config;
    std::string _configPath;

    std::string _scenarioType;
    bool _refinementEnabled;
    std::string _refinementCriterion;

    float _cooldownTime;
    float _lastSampleTime = 0.0F;
};

}
