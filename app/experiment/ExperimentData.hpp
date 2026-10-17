#pragma once

#include <cstdint>
#include <glm/ext/vector_float3.hpp>
#include <string>
#include <vector>

namespace sph::experiment
{

struct PerFrameMetrics
{
    uint32_t frame;
    uint32_t iteration;
    float simulationTime;
    float computationTimeMs;
    uint32_t particleCount;
    float rmseDensity;
};

struct FinalState
{
    uint32_t frame;
    uint32_t iteration;
    float simulationTime;
    uint32_t particleCount;
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> velocities;
};

struct ExperimentMetadata
{
    std::string experimentId;
    std::string name;
    std::string timestamp;
    std::string configurationFile;
};

struct StatisticalSummary
{
    uint32_t totalFrames;
    uint32_t totalIterations;
    float totalSimulationTime;
    float totalComputationTimeMs;
    float averageFrameTimeMs;
    float minFrameTimeMs;
    float maxFrameTimeMs;
    uint32_t particleCountMin;
    uint32_t particleCountMax;
    uint32_t particleCountFinal;
    float rmseDensityMean;
    float rmseDensityMax;
};

struct ExperimentData
{
    ExperimentMetadata metadata;
    std::vector<PerFrameMetrics> perFrameMetrics;
    FinalState finalState;
    StatisticalSummary summary;
};

}
