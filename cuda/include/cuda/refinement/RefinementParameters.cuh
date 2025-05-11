#pragma once

#include <cstdint>
#include <string>

namespace sph::cuda::refinement
{

struct VelocityParameters
{
    struct Split
    {
        float minimalSpeedThreshold = 2.0f;
    };

    struct Merge
    {
        float maximalSpeedThreshold = 0.1f;
    };

    Split split;
    Merge merge;
};

struct VorticityParameters
{
    struct Split
    {
        float minimalVorticityThreshold = 1.0f;
    };

    struct Merge
    {
        float maximalVorticityThreshold = 0.3f;
    };

    Split split;
    Merge merge;
};

struct CurvatureParameters
{
    struct Split
    {
        float minimalCurvatureThreshold = 2.0f;
    };

    struct Merge
    {
        float maximalCurvatureThreshold = 0.6f;
    };

    Split split;
    Merge merge;
};

struct InterfaceParameters
{
    struct Split
    {
        float distanceRatioThreshold = 0.05f;  // Fraction of domain size
    };

    struct Merge
    {
        float distanceRatioThreshold = 0.15f;  // Fraction of domain size
    };

    Split split;
    Merge merge;
};

struct SplittingParameters
{
    float epsilon = 0.65f;           // Spacing parameter for daughter particles (from Vacondio 2016)
    float alpha = 0.70f;             // Smoothing length ratio for daughter particles
    float centerMassRatio = 0.077f;  // Mass ratio for center particle
    float vertexMassRatio = 0.077F;  // Mass ratio for each vertex particle (12 vertices)
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
