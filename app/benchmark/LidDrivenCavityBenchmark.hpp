// LidDrivenCavityBenchmark.hpp
#pragma once
#include <glm/glm.hpp>
#include <vector>

#include "BenchmarkCase.hpp"

namespace sph
{

class LidDrivenCavityBenchmark : public BenchmarkCase
{
public:
    void initialize(cuda::Simulation& simulation, const nlohmann::json& config) override
    {
        // Extract parameters
        float Re = config.value("reynoldsNumber", 100.0f);
        float cavitySize = config.value("cavitySize", 1.0f);
        float particleSpacing = config.value("particleSpacing", 0.025f);

        // Calculate viscosity based on Reynolds number
        float lidVelocity = 1.0f;  // Unity lid velocity
        float kinematicViscosity = lidVelocity * cavitySize / Re;

        // Set up simulation parameters
        auto& params = simulation.getParameters();
        params.domain.min = glm::vec3(0.0f);
        params.domain.max = glm::vec3(cavitySize);
        params.viscosityConstant = kinematicViscosity;
        params.gravity = glm::vec3(0.0f);  // No gravity
        params.baseSmoothingRadius = 2.5f * particleSpacing;
        params.baseParticleRadius = particleSpacing / 2.0f;

        // Initialize particle positions in regular grid
        std::vector<glm::vec4> positions;

        for (float x = particleSpacing / 2; x < cavitySize; x += particleSpacing)
        {
            for (float y = particleSpacing / 2; y < cavitySize; y += particleSpacing)
            {
                for (float z = particleSpacing / 2; z < cavitySize; z += particleSpacing)
                {
                    positions.push_back(glm::vec4(x, y, z, 0.0f));
                }
            }
        }

        simulation.setParticles(positions);

        _lidVelocity = lidVelocity;
        _particleRadius = params.baseParticleRadius;
    }

    void applyBoundaryConditions(cuda::Simulation& simulation, float time) override
    {
        // Apply lid velocity to particles near top boundary
        auto particles = simulation.getParticles();
        const float cavitySize = simulation.getParameters().domain.max.y;
        const float boundaryLayer = 2.0f * _particleRadius;

        for (uint32_t i = 0; i < particles.particleCount; i++)
        {
            const float x = particles.positions[i].x;
            const float y = particles.positions[i].y;
            const float z = particles.positions[i].z;

            // Particles near top boundary get lid velocity
            if (y > cavitySize - boundaryLayer)
            {
                particles.velocities[i] = glm::vec4(_lidVelocity, 0.0f, 0.0f, 0.0f);
            }
            // No-slip on other walls
            else if (y < boundaryLayer || x < boundaryLayer || x > cavitySize - boundaryLayer || z < boundaryLayer ||
                     z > cavitySize - boundaryLayer)
            {
                particles.velocities[i] = glm::vec4(0.0f);
            }
        }
    }

    BenchmarkResults analyze(const cuda::Simulation& simulation) override
    {
        BenchmarkResults results;

        auto particles = simulation.getParticles();
        const float cavitySize = simulation.getParameters().domain.max.x;

        // Extract velocity profiles along centerlines
        const float center = cavitySize / 2.0f;
        const float tolerance = simulation.getParameters().baseSmoothingRadius;

        // Vertical centerline (x = L/2, z = L/2)
        std::vector<std::pair<float, float>> vxProfile;
        // Horizontal centerline (y = L/2, z = L/2)
        std::vector<std::pair<float, float>> vyProfile;

        for (uint32_t i = 0; i < particles.particleCount; i++)
        {
            const glm::vec3 pos(particles.positions[i]);
            const glm::vec3 vel(particles.velocities[i]);

            // Check if on vertical centerline
            if (std::abs(pos.x - center) < tolerance && std::abs(pos.z - center) < tolerance)
            {
                vxProfile.push_back({pos.y / cavitySize, vel.x});
            }

            // Check if on horizontal centerline
            if (std::abs(pos.y - center) < tolerance && std::abs(pos.z - center) < tolerance)
            {
                vyProfile.push_back({pos.x / cavitySize, vel.y});
            }
        }

        // Sort profiles by position
        std::sort(vxProfile.begin(), vxProfile.end());
        std::sort(vyProfile.begin(), vyProfile.end());

        // Convert to results format
        for (const auto& point : vxProfile)
        {
            results.vxProfile.push_back(glm::vec2(point.first, point.second));
        }
        for (const auto& point : vyProfile)
        {
            results.vyProfile.push_back(glm::vec2(point.first, point.second));
        }

        // Find primary vortex center
        glm::vec2 vortexCenter = findPrimaryVortexCenter(particles, cavitySize);
        results.vortexCenterX = vortexCenter.x;
        results.vortexCenterY = vortexCenter.y;

        // Calculate total kinetic energy
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
    float _lidVelocity = 1.0f;
    float _particleRadius = 0.0125f;

    glm::vec2 findPrimaryVortexCenter(const cuda::ParticlesData& particles, float cavitySize)
    {
        // Find center of primary vortex by looking for minimum velocity
        float minVelocity = FLT_MAX;
        glm::vec2 vortexCenter(0.5f, 0.5f);  // Default to center

        const float centerRegionMin = 0.3f * cavitySize;
        const float centerRegionMax = 0.7f * cavitySize;

        for (uint32_t i = 0; i < particles.particleCount; i++)
        {
            const glm::vec3 pos(particles.positions[i]);
            const glm::vec3 vel(particles.velocities[i]);

            // Check if in central region and mid-depth slice
            if (pos.x > centerRegionMin && pos.x < centerRegionMax && pos.y > centerRegionMin &&
                pos.y < centerRegionMax && std::abs(pos.z - cavitySize / 2) < 0.1f * cavitySize)
            {
                float velMag = glm::length(glm::vec2(vel.x, vel.y));

                if (velMag < minVelocity)
                {
                    minVelocity = velMag;
                    vortexCenter = glm::vec2(pos.x / cavitySize, pos.y / cavitySize);
                }
            }
        }

        return vortexCenter;
    }
};

}
