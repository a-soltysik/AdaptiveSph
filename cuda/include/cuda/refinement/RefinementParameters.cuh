#pragma once

#include <cstdint>

namespace sph::cuda::refinement
{

struct VelocityParameters
{
    struct Split
    {
        float minimalSpeedThreshold = 2.0F;
    };

    struct Merge
    {
        float maximalSpeedThreshold = 0.1F;
    };

    Split split;
    Merge merge;
};

struct VorticityParameters
{
    struct Split
    {
        float minimalVorticityThreshold = 1.0F;
    };

    struct Merge
    {
        float maximalVorticityThreshold = 0.3F;
    };

    Split split;
    Merge merge;
};

struct InterfaceParameters
{
    struct Split
    {
        float distanceRatioThreshold = 0.05F;  // Fraction of domain size
    };

    struct Merge
    {
        float distanceRatioThreshold = 0.15F;  // Fraction of domain size
    };

    Split split;
    Merge merge;
};

struct SplittingParameters
{
    float epsilon = 0.65F;           // Spacing parameter for daughter particles (from Vacondio 2016)
    float alpha = 0.70F;             // Smoothing length ratio for daughter particles
    float centerMassRatio = 0.077F;  // Mass ratio for center particle
    float vertexMassRatio = 0.077F;  // Mass ratio for each vertex particle (12 vertices)
};

struct RefinementParameters
{
    enum class Criterion
    {
        Velocity,
        Interface,
        Vorticity
    };

    bool enabled = false;
    float minMassRatio = 0.3F;
    float maxMassRatio = 8.0F;
    uint32_t maxParticleCount = 100000000;
    float maxBatchRatio = 0.2F;
    uint32_t initialCooldown = 10;
    uint32_t cooldown = 10;
    Criterion criterionType = Criterion::Velocity;  // Added criterion type selector

    SplittingParameters splitting;
    VelocityParameters velocity;
    VorticityParameters vorticity;
    InterfaceParameters interfaceParameters;
};

}
