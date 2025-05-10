// TaylorGreenBenchmark.hpp
#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <vector>

#include "BenchmarkCase.hpp"

namespace sph
{

class TaylorGreenBenchmark : public BenchmarkCase
{
public:
    void initialize(Simulation& simulation, const nlohmann::json& config) override
    {
        // Extract parameters
        float Re = config.value("reynoldsNumber", 100.0f);
        float domainSize = config.value("domainSize", 1.0f);
        float particleSpacing = config.value("particleSpacing", 0.025f);

        // Calculate viscosity from Reynolds number
        float U0 = 1.0f;  // Reference velocity
        float kinematicViscosity = U0 * domainSize / Re;
        _decay_rate = 8.0f * glm::pi<float>() * glm::pi<float>() * kinematicViscosity / (domainSize * domainSize);

        // Set up simulation parameters
        auto& params = simulation.getParameters();
        params.domain.min = glm::vec3(0.0f);
        params.domain.max = glm::vec3(domainSize);
        params.viscosityConstant = kinematicViscosity;
        params.gravity = glm::vec3(0.0f);  // No gravity
        params.baseSmoothingRadius = 2.5f * particleSpacing;
        params.baseParticleRadius = particleSpacing / 2.0f;

        // Initialize particles with Taylor-Green velocity field
        std::vector<glm::vec4> positions;
        std::vector<glm::vec4> velocities;

        for (float x = particleSpacing / 2; x < domainSize; x += particleSpacing)
        {
            for (float y = particleSpacing / 2; y < domainSize; y += particleSpacing)
            {
                for (float z = particleSpacing / 2; z < domainSize; z += particleSpacing)
                {
                    positions.push_back(glm::vec4(x, y, z, 0.0f));

                    // Taylor-Green initial velocity field
                    float vx = U0 * std::sin(2.0f * glm::pi<float>() * x / domainSize) *
                               std::cos(2.0f * glm::pi<float>() * y / domainSize);
                    float vy = -U0 * std::cos(2.0f * glm::pi<float>() * x / domainSize) *
                               std::sin(2.0f * glm::pi<float>() * y / domainSize);
                    float vz = 0.0f;

                    velocities.push_back(glm::vec4(vx, vy, vz, 0.0f));
                }
            }
        }

        simulation.setParticles(positions, velocities);
    }

    void applyBoundaryConditions(Simulation& simulation, float time) override
    {
        // Taylor-Green vortex uses periodic boundary conditions
        auto particles = simulation.getParticles();
        const float domainSize = simulation.getParameters().domain.max.x;

        for (uint32_t i = 0; i < particles.particleCount; i++)
        {
            // Apply periodic boundary conditions in all directions
            for (int d = 0; d < 3; d++)
            {
                if (particles.positions[i][d] < 0)
                {
                    particles.positions[i][d] += domainSize;
                }
                else if (particles.positions[i][d] > domainSize)
                {
                    particles.positions[i][d] -= domainSize;
                }
            }
        }
    }

    BenchmarkResults analyze(const Simulation& simulation) override
    {
        BenchmarkResults results;

        auto particles = simulation.getParticles();
        const float domainSize = simulation.getParameters().domain.max.x;
        const float currentTime = simulation.getCurrentTime();

        // Calculate total kinetic energy
        float kineticEnergy = 0.0f;
        float totalError = 0.0f;
        int sampleCount = 0;

        // Sample points on a grid to compare with analytical solution
        const int gridSize = 20;
        const float dx = domainSize / gridSize;

        for (int i = 0; i < gridSize; i++)
        {
            for (int j = 0; j < gridSize; j++)
            {
                float x = (i + 0.5f) * dx;
                float y = (j + 0.5f) * dx;
                float z = domainSize / 2.0f;  // Sample at mid-plane

                // Find nearest particle
                float minDist = FLT_MAX;
                glm::vec3 numerical_velocity(0.0f);

                for (uint32_t p = 0; p < particles.particleCount; p++)
                {
                    glm::vec3 pos(particles.positions[p]);
                    float dist = glm::distance(pos, glm::vec3(x, y, z));

                    if (dist < minDist)
                    {
                        minDist = dist;
                        numerical_velocity = glm::vec3(particles.velocities[p]);
                    }
                }

                // Calculate analytical solution at this point
                float decay_factor = std::exp(-_decay_rate * currentTime);
                float vx_analytical = decay_factor * std::sin(2.0f * glm::pi<float>() * x / domainSize) *
                                      std::cos(2.0f * glm::pi<float>() * y / domainSize);
                float vy_analytical = -decay_factor * std::cos(2.0f * glm::pi<float>() * x / domainSize) *
                                      std::sin(2.0f * glm::pi<float>() * y / domainSize);

                glm::vec2 analytical_velocity(vx_analytical, vy_analytical);

                // Calculate error
                float error = glm::distance(glm::vec2(numerical_velocity), analytical_velocity);
                totalError += error * error;
                sampleCount++;
            }
        }

        // Calculate total kinetic energy
        for (uint32_t i = 0; i < particles.particleCount; i++)
        {
            glm::vec3 vel(particles.velocities[i]);
            kineticEnergy += 0.5f * particles.masses[i] * glm::dot(vel, vel);
        }

        results.kineticEnergy = kineticEnergy;
        results.velocityErrors.push_back(std::sqrt(totalError / sampleCount));

        return results;
    }

private:
    float _decay_rate = 0.0f;
};

}
