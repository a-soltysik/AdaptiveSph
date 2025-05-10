// PoiseuilleBenchmark.hpp
#pragma once
#include <glm/glm.hpp>
#include <vector>

#include "BenchmarkCase.hpp"

namespace sph
{

class PoiseuilleBenchmark : public BenchmarkCase
{
public:
    void initialize(cuda::Simulation& simulation, const nlohmann::json& config) override
    {
        // Extract parameters from config
        float Re = config.value("reynoldsNumber", 30.0f);
        float channelHeight = config.value("channelHeight", 0.1f);
        float channelLength = config.value("channelLength", 0.5f);
        float channelWidth = config.value("channelWidth", 0.1f);
        float particleSpacing = config.value("particleSpacing", 0.0025f);

        // Calculate viscosity from Reynolds number
        float maxVelocity = 1.0f;  // Maximum velocity in channel
        float kinematicViscosity = maxVelocity * channelHeight / Re;

        // Set up simulation parameters
        auto& params = simulation.getParameters();
        params.domain.min = glm::vec3(0.0f, 0.0f, 0.0f);
        params.domain.max = glm::vec3(channelLength, channelHeight, channelWidth);
        params.viscosityConstant = kinematicViscosity;
        params.gravity = glm::vec3(0.0f, 0.0f, 0.0f);  // No gravity
        params.baseSmoothingRadius = 2.5f * particleSpacing;
        params.baseParticleRadius = particleSpacing / 2.0f;

        // Calculate pressure gradient to drive flow
        // For Poiseuille flow: dp/dx = -8μU_max/H²
        float pressureGradient =
            -8.0f * params.restDensity * kinematicViscosity * maxVelocity / (channelHeight * channelHeight);
        _pressureGradient = pressureGradient;

        // Initialize particle positions in a regular grid
        std::vector<glm::vec4> positions;
        for (float x = particleSpacing / 2; x < channelLength; x += particleSpacing)
        {
            for (float y = particleSpacing / 2; y < channelHeight; y += particleSpacing)
            {
                for (float z = particleSpacing / 2; z < channelWidth; z += particleSpacing)
                {
                    positions.push_back(glm::vec4(x, y, z, 0.0f));
                }
            }
        }

        simulation.setParticles(positions);
    }

    void applyBoundaryConditions(cuda::Simulation& simulation, float time) override
    {
        // Apply periodic boundary conditions in x-direction
        auto particles = simulation.getParticles();
        const float channelLength = simulation.getParameters().domain.max.x;

        for (uint32_t i = 0; i < particles.particleCount; i++)
        {
            // Periodic BC in x-direction
            if (particles.positions[i].x < 0)
            {
                particles.positions[i].x += channelLength;
            }
            else if (particles.positions[i].x > channelLength)
            {
                particles.positions[i].x -= channelLength;
            }

            // Apply pressure gradient as body force
            particles.forces[i].x += _pressureGradient / simulation.getParameters().restDensity;
        }
    }

    BenchmarkResults analyze(const cuda::Simulation& simulation) override
    {
        BenchmarkResults results;

        auto particles = simulation.getParticles();
        const auto& params = simulation.getParameters();
        const float channelHeight = params.domain.max.y;

        // Extract velocity profile at channel center (x = L/2)
        const float centerX = params.domain.max.x / 2.0f;
        const float tolerance = params.baseSmoothingRadius;

        std::vector<std::pair<float, float>> velocityProfile;

        for (uint32_t i = 0; i < particles.particleCount; i++)
        {
            // Check if particle is near the center slice
            if (std::abs(particles.positions[i].x - centerX) < tolerance)
            {
                float y = particles.positions[i].y;
                float vx = particles.velocities[i].x;
                velocityProfile.push_back({y / channelHeight, vx});
            }
        }

        // Sort by y-coordinate
        std::sort(velocityProfile.begin(), velocityProfile.end());

        // Convert to required format
        for (const auto& point : velocityProfile)
        {
            results.vxProfile.push_back(glm::vec2(point.first, point.second));
        }

        // Calculate analytical solution and error
        float maxVelocity = calculateMaxVelocity(channelHeight);
        float totalError = 0.0f;

        for (const auto& point : velocityProfile)
        {
            float y_normalized = point.first;
            float analytical = 4.0f * maxVelocity * y_normalized * (1.0f - y_normalized);
            float numerical = point.second;
            totalError += std::pow(numerical - analytical, 2.0f);
        }

        results.velocityErrors.push_back(std::sqrt(totalError / velocityProfile.size()));

        return results;
    }

private:
    float _pressureGradient = 0.0f;

    float calculateMaxVelocity(float channelHeight)
    {
        // U_max = -dp/dx * H²/(8μ)
        const float viscosity = 0.01f;  // Should match config
        return std::abs(_pressureGradient) * channelHeight * channelHeight / (8.0f * viscosity);
    }
};

}
