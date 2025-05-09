#pragma once

#include <cstdint>

#include "VelocityParameters.cuh"

namespace sph::cuda::refinement
{

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

    SplittingParameters splitting;
    VelocityParameters velocity;
};

}
