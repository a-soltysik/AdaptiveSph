// DamBreakBenchmark.hpp
#pragma once
#include <glm/glm.hpp>
#include <vector>

#include "BenchmarkCase.hpp"

namespace sph
{

class DamBreakBenchmark : public BenchmarkCase
{
public:
    void initialize(cuda::Simulation& simulation, const nlohmann::json& config) override
    {
        // Extract parameters
        float tankLength = config.value("tankLength", 4.0f);
        float tankHeight = config.value("tankHeight", 2.0f);
        float tankWidth = config.value("tankWidth", 1.0f);
        float waterColumnWidth = config.value("waterColumnWidth", 1.0f);
        float waterColumnHeight = config.value("waterColumnHeight", 1.0f);
        float particleSpacing = config.value("particleSpacing", 0.025f);
        // Set up simulation parameters
        auto& params = simulation.getParameters();
        params.domain.min = glm::vec3(0.0f);
        params.domain.max = glm::vec3(tankLength, tankHeight, tankWidth);
        params.gravity = glm::vec3(0.0f, -9.81f, 0.0f);
        params.baseSmoothingRadius = 2.5f * particleSpacing;
        params.baseParticleRadius = particleSpacing / 2.0f;
        // Initialize fluid particles in water column
        std::vector<glm::vec4> positions;
        for (float x = particleSpacing / 2; x < waterColumnWidth; x += particleSpacing)
        {
            for (float y = particleSpacing / 2; y < waterColumnHeight; y += particleSpacing)
            {
                for (float z = particleSpacing / 2; z < tankWidth; z += particleSpacing)
                {
                    positions.push_back(glm::vec4(x, y, z, 0.0f));
                }
            }
        }
        simulation.setParticles(positions);
        // Set up gauge positions for measuring water height
        _gaugePositions = {
            glm::vec2(0.5f * waterColumnWidth, 0.0f),  // Inside initial column
            glm::vec2(1.5f * waterColumnWidth, 0.0f),  // Just past column
            glm::vec2(3.0f * waterColumnWidth, 0.0f)   // Far from column
        };
    }

    void applyBoundaryConditions(cuda::Simulation& simulation, float time) override
    {
        // No special boundary conditions needed - walls are handled by domain boundaries
    }

    BenchmarkResults analyze(const cuda::Simulation& simulation) override
    {
        BenchmarkResults results;
        auto particles = simulation.getParticles();
        // Find leading edge position (rightmost fluid particle)
        float leadingEdge = 0.0f;
        for (uint32_t i = 0; i < particles.particleCount; i++)
        {
            leadingEdge = std::max(leadingEdge, particles.positions[i].x);
        }
        results.leadingEdgePosition = leadingEdge;
        // Measure height at gauge positions
        results.gaugeHeights.clear();
        const float gaugeRadius = 0.1f;  // Measurement radius around gauge
        for (const auto& gaugePos : _gaugePositions)
        {
            float maxHeight = 0.0f;
            for (uint32_t i = 0; i < particles.particleCount; i++)
            {
                float x = particles.positions[i].x;
                float y = particles.positions[i].y;
                // Check if particle is near this gauge position
                if (std::abs(x - gaugePos.x) < gaugeRadius)
                {
                    maxHeight = std::max(maxHeight, y);
                }
            }
            results.gaugeHeights.push_back(maxHeight);
        }
        // Calculate total fluid kinetic energy
        float kineticEnergy = 0.0f;
        for (uint32_t i = 0; i < particles.particleCount; i++)
        {
            glm::vec3 vel(particles.velocities[i]);
            kineticEnergy += 0.5f * particles.masses[i] * glm::dot(vel, vel);
        }
        results.kineticEnergy = kineticEnergy;

        return results;
    }

private:
    std::vector<glm::vec2> _gaugePositions;
};

}
