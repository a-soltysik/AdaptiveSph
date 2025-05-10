#pragma once

#include <cstdint>
#include <string>

namespace sph::cuda::refinement
{

struct VelocityParameters
{
    struct Split
    {
        float minimalSpeedThreshold;
    };

    struct Merge
    {
        float maximalSpeedThreshold;
    };

    Split split;
    Merge merge;
    uint64_t dummy {};
};

struct SplittingParameters
{
    float epsilon = 0.65f;           // Spacing parameter for daughter particles (from Vacondio 2016)
    float alpha = 0.70f;             // Smoothing length ratio for daughter particles
    float centerMassRatio = 0.077f;  // Mass ratio for center particle
    float vertexMassRatio = 0.077F;  // Mass ratio for each vertex particle (12 vertices)
};

struct InterfaceParameters
{
    float splittingThresholdRatio = 0.05f;  // Fraction of domain size for splitting zone
    float mergingThresholdRatio = 0.15f;    // Fraction of domain size for merging zone
    uint64_t dummy {};
};

struct VorticityParameters
{
    float threshold = 1.0f;    // Minimum vorticity magnitude for refinement
    float scaleFactor = 1.0f;  // Scaling factor for vorticity-based refinement
    uint64_t dummy {};
};

struct CurvatureParameters
{
    float threshold = 2.0f;    // Minimum curvature magnitude for refinement
    float scaleFactor = 1.0f;  // Scaling factor for curvature-based refinement
    uint64_t dummy {};
};

struct RefinementParameters
{
    bool enabled = false;
    float minMassRatio = 0.3f;
    float maxMassRatio = 8.0f;
    uint32_t maxParticleCount = 500000;
    float maxBatchRatio = 0.2F;
    uint32_t initialCooldown = 10;
    uint32_t cooldown = 10;
    std::string criterionType = "velocity";  // Added criterion type selector

    SplittingParameters splitting;
    VelocityParameters velocity;

    VorticityParameters vorticity;
    CurvatureParameters curvature;
    InterfaceParameters interfaceParameters;
};

}
