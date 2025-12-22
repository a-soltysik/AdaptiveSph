#include "ExperimentDataCollector.hpp"

#include <panda/Logger.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <sstream>

#include "utils/Configuration.hpp"

using json = nlohmann::json;

namespace sph::experiment
{

ExperimentDataCollector::ExperimentDataCollector(cuda::Simulation* simulation,
                                                 const std::string& configPath,
                                                 const utils::ExperimentConfig& config,
                                                 const std::string& scenarioType,
                                                 bool refinementEnabled,
                                                 const std::string& refinementCriterion,
                                                 float cooldownTime)
    : _simulation(simulation),
      _config(config),
      _configPath(configPath),
      _scenarioType(scenarioType),
      _refinementEnabled(refinementEnabled),
      _refinementCriterion(refinementCriterion),
      _cooldownTime(cooldownTime)
{
    if (!_config.enabled)
    {
        return;
    }

    _data.metadata.timestamp = getCurrentTimestamp();
    _data.metadata.experimentId = _data.metadata.timestamp;
    _data.metadata.name = generateExperimentName();
    _data.metadata.configurationFile = configPath;

    // Initialize last sample time to cooldown (so first sample is at cooldown time)
    _lastSampleTime = _cooldownTime;

    panda::log::Info("Experiment data collection enabled: {}", _data.metadata.name);
    panda::log::Info("Cooldown time: {:.2f}s (experiment time starts after cooldown)", _cooldownTime);
    panda::log::Info("Output will be saved to: {}", generateOutputPath());
}

void ExperimentDataCollector::onFrameCompleted(float computationTimeMs)
{
    if (!_config.enabled)
    {
        return;
    }

    const float currentTime = _simulation->getCurrentSimulationTime();

    // Only collect data after cooldown period
    if (currentTime < _cooldownTime)
    {
        return;
    }

    if (currentTime - _lastSampleTime >= _config.samplingInterval)
    {
        collectPerFrameMetrics(computationTimeMs);
        _lastSampleTime = currentTime;
    }
}

void ExperimentDataCollector::finalize()
{
    if (!_config.enabled)
    {
        return;
    }

    panda::log::Info("Finalizing experiment data collection...");

    collectFinalState();
    calculateStatistics();
    saveToJson();

    panda::log::Info("Experiment data saved to: {}", generateOutputPath());
}

void ExperimentDataCollector::collectPerFrameMetrics(float computationTimeMs)
{
    PerFrameMetrics metrics;
    metrics.frame = static_cast<uint32_t>(_data.perFrameMetrics.size());
    metrics.iteration = _simulation->getCurrentIteration();
    // Subtract cooldown time so experiment time starts at 0
    metrics.simulationTime = _simulation->getCurrentSimulationTime() - _cooldownTime;
    metrics.computationTimeMs = computationTimeMs;
    metrics.particleCount = _simulation->getFluidParticlesCount();
    metrics.rmseDensity = _simulation->calculateDensityRMSE();

    _data.perFrameMetrics.push_back(metrics);

    panda::log::Debug("Collected metrics at experiment t={:.2f}s: {} particles, RMSE density={:.4f}",
                      metrics.simulationTime,
                      metrics.particleCount,
                      metrics.rmseDensity);
}

void ExperimentDataCollector::collectFinalState()
{
    _data.finalState.frame = static_cast<uint32_t>(_data.perFrameMetrics.size());
    _data.finalState.iteration = _simulation->getCurrentIteration();
    // Subtract cooldown time so experiment time starts at 0
    _data.finalState.simulationTime = _simulation->getCurrentSimulationTime() - _cooldownTime;
    _data.finalState.particleCount = _simulation->getFluidParticlesCount();

    _simulation->copyParticlePositionsToHost(_data.finalState.positions);
    _simulation->copyParticleVelocitiesToHost(_data.finalState.velocities);

    panda::log::Info("Copied {} particles from GPU to host", _data.finalState.particleCount);
}

void ExperimentDataCollector::calculateStatistics()
{
    if (_data.perFrameMetrics.empty())
    {
        return;
    }

    _data.summary.totalFrames = static_cast<uint32_t>(_data.perFrameMetrics.size());
    _data.summary.totalIterations = _data.finalState.iteration;
    _data.summary.totalSimulationTime = _data.finalState.simulationTime;

    _data.summary.totalComputationTimeMs = 0.0F;
    _data.summary.minFrameTimeMs = std::numeric_limits<float>::max();
    _data.summary.maxFrameTimeMs = 0.0F;
    _data.summary.particleCountMin = std::numeric_limits<uint32_t>::max();
    _data.summary.particleCountMax = 0;
    _data.summary.rmseDensityMean = 0.0F;
    _data.summary.rmseDensityMax = 0.0F;

    uint32_t validRmseDensityCount = 0;

    for (const auto& metrics : _data.perFrameMetrics)
    {
        _data.summary.totalComputationTimeMs += metrics.computationTimeMs;
        _data.summary.minFrameTimeMs = std::min(_data.summary.minFrameTimeMs, metrics.computationTimeMs);
        _data.summary.maxFrameTimeMs = std::max(_data.summary.maxFrameTimeMs, metrics.computationTimeMs);
        _data.summary.particleCountMin = std::min(_data.summary.particleCountMin, metrics.particleCount);
        _data.summary.particleCountMax = std::max(_data.summary.particleCountMax, metrics.particleCount);

        if (std::isfinite(metrics.rmseDensity))
        {
            _data.summary.rmseDensityMean += metrics.rmseDensity;
            _data.summary.rmseDensityMax = std::max(_data.summary.rmseDensityMax, metrics.rmseDensity);
            ++validRmseDensityCount;
        }
    }

    _data.summary.averageFrameTimeMs =
        _data.summary.totalComputationTimeMs / static_cast<float>(_data.summary.totalFrames);
    if (validRmseDensityCount > 0)
    {
        _data.summary.rmseDensityMean /= static_cast<float>(validRmseDensityCount);
    }
    else
    {
        _data.summary.rmseDensityMean = 0.0F;
    }
    _data.summary.particleCountFinal = _data.finalState.particleCount;
}

void ExperimentDataCollector::saveToJson()
{
    json j;

    j["experiment_metadata"] = {
        {"experiment_id",      _data.metadata.experimentId     },
        {"name",               _data.metadata.name             },
        {"timestamp",          _data.metadata.timestamp        },
        {"configuration_file", _data.metadata.configurationFile}
    };

    j["per_frame_metrics"] = json::array();
    for (const auto& metrics : _data.perFrameMetrics)
    {
        j["per_frame_metrics"].push_back({
            {"frame",               metrics.frame            },
            {"iteration",           metrics.iteration        },
            {"simulation_time",     metrics.simulationTime   },
            {"computation_time_ms", metrics.computationTimeMs},
            {"particle_count",      metrics.particleCount    },
            {"rmse_density",        metrics.rmseDensity      }
        });
    }

    j["final_state"] = {
        {"frame",           _data.finalState.frame         },
        {"iteration",       _data.finalState.iteration     },
        {"simulation_time", _data.finalState.simulationTime},
        {"particle_count",  _data.finalState.particleCount }
    };

    j["final_state"]["particles"]["positions"] = json::array();
    for (const auto& pos : _data.finalState.positions)
    {
        j["final_state"]["particles"]["positions"].push_back({pos.x, pos.y, pos.z});
    }

    j["final_state"]["particles"]["velocities"] = json::array();
    for (const auto& vel : _data.finalState.velocities)
    {
        j["final_state"]["particles"]["velocities"].push_back({vel.x, vel.y, vel.z});
    }

    j["statistical_summary"] = {
        {"total_frames",              _data.summary.totalFrames           },
        {"total_iterations",          _data.summary.totalIterations       },
        {"total_simulation_time",     _data.summary.totalSimulationTime   },
        {"total_computation_time_ms", _data.summary.totalComputationTimeMs},
        {"average_frame_time_ms",     _data.summary.averageFrameTimeMs    },
        {"min_frame_time_ms",         _data.summary.minFrameTimeMs        },
        {"max_frame_time_ms",         _data.summary.maxFrameTimeMs        },
        {"particle_count_min",        _data.summary.particleCountMin      },
        {"particle_count_max",        _data.summary.particleCountMax      },
        {"particle_count_final",      _data.summary.particleCountFinal    },
        {"rmse_density_mean",         _data.summary.rmseDensityMean       },
        {"rmse_density_max",          _data.summary.rmseDensityMax        }
    };

    const auto outputPath = generateOutputPath();
    std::filesystem::create_directories(std::filesystem::path(outputPath).parent_path());

    std::ofstream file(outputPath);
    if (!file.is_open())
    {
        panda::log::Error("Failed to open file for writing: {}", outputPath);
        return;
    }

    file << j.dump(4);
    file.close();
}

auto ExperimentDataCollector::generateOutputPath() const -> std::string
{
    return "experiments/experiment_" + _data.metadata.timestamp + "_" + _data.metadata.name + ".json";
}

auto ExperimentDataCollector::generateExperimentName() const -> std::string
{
    std::string name = _scenarioType;
    if (_refinementEnabled)
    {
        name += "_adaptive_" + _refinementCriterion;
    }
    else
    {
        name += "_standard";
    }
    return name;
}

auto ExperimentDataCollector::getCurrentTimestamp() const -> std::string
{
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d_%H-%M-%S");
    return ss.str();
}

}
