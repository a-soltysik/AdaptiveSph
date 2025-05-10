// VorticityCriterion.cu
#include <cfloat>
#include <glm/geometric.hpp>

#include "../common/Iteration.hpp"
#include "../common/Utils.cuh"
#include "../device/Kernel.cuh"
#include "VorticityCriterion.cuh"

namespace sph::cuda::refinement::vorticity
{

__device__ auto SplitCriterionGenerator::operator()(ParticlesData particles,
                                                    uint32_t id,
                                                    const SphSimulation::Grid& grid,
                                                    const Simulation::Parameters& simulationData) const -> float
{
    if (particles.masses[id] < _minimalMass)
    {
        return FLT_MAX;
    }
    // Calculate vorticity using finite differences
    const glm::vec4 position = particles.positions[id];
    const glm::vec4 velocity = particles.velocities[id];
    const float h = particles.smoothingRadiuses[id];
    // Initialize velocity gradient components
    float dvx_dy = 0.0f, dvx_dz = 0.0f;
    float dvy_dx = 0.0f, dvy_dz = 0.0f;
    float dvz_dx = 0.0f, dvz_dy = 0.0f;
    int validNeighbors = 0;
    // Use spatial hash grid for efficient neighbor search
    forEachNeighbour(position, simulationData, grid, [&](const auto neighbourIdx) {
        if (neighbourIdx == id)
        {
            return;
        }
        const glm::vec4 neighborPos = particles.positions[neighbourIdx];
        const glm::vec3 r = glm::vec3(neighborPos - position);
        const float dist = glm::length(r);
        if (dist < h && dist > 1e-5f)
        {
            const glm::vec4 neighborVel = particles.velocities[neighbourIdx];
            const glm::vec3 velDiff = glm::vec3(neighborVel - velocity);
            const glm::vec3 dir = r / dist;
            // Kernel gradient (using density derivative kernel)
            const float gradW = device::densityDerivativeKernel(dist, h);
            const float weight = particles.masses[neighbourIdx] / particles.densities[neighbourIdx] * gradW;
            // Accumulate velocity gradients
            dvx_dy += weight * dir.y * velDiff.x;
            dvx_dz += weight * dir.z * velDiff.x;
            dvy_dx += weight * dir.x * velDiff.y;
            dvy_dz += weight * dir.z * velDiff.y;
            dvz_dx += weight * dir.x * velDiff.z;
            dvz_dy += weight * dir.y * velDiff.z;
            validNeighbors++;
        }
    });
    if (validNeighbors < 3)
    {
        return FLT_MAX;  // Not enough neighbors for reliable vorticity
    }
    // Calculate vorticity components: ω = ∇ × v
    float omega_x = dvz_dy - dvy_dz;
    float omega_y = dvx_dz - dvz_dx;
    float omega_z = dvy_dx - dvx_dy;
    float vorticityMagnitude = glm::length(glm::vec3(omega_x, omega_y, omega_z));
    if (vorticityMagnitude > _vorticity.threshold)
    {
        return vorticityMagnitude * _vorticity.scaleFactor;
    }

    return FLT_MAX;
}

__device__ auto MergeCriterionGenerator::operator()(ParticlesData particles,
                                                    uint32_t id,
                                                    const SphSimulation::Grid& grid,
                                                    const Simulation::Parameters& simulationData) const -> float
{
    if (particles.masses[id] > _maximalMass)
    {
        return FLT_MAX;
    }

    // Calculate vorticity to check if it's low enough for merging
    SplitCriterionGenerator splitGen(0.0f, _vorticity);
    float splitValue = splitGen(particles, id, grid, simulationData);

    // Use lower threshold for merging
    if (splitValue != FLT_MAX && splitValue > _vorticity.threshold * 0.5f)
    {
        return FLT_MAX;  // High vorticity, don't merge
    }

    return splitValue == FLT_MAX ? 0.0f : splitValue;
}

}
