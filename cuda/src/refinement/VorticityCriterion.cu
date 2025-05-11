#include <cfloat>
#include <glm/geometric.hpp>

#include "../common/Iteration.hpp"
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
        return -1;
    }

    const glm::vec4 position = particles.positions[id];
    const float h = particles.smoothingRadiuses[id];

    float dvx_dx = 0.0f, dvx_dy = 0.0f, dvx_dz = 0.0f;
    float dvy_dx = 0.0f, dvy_dy = 0.0f, dvy_dz = 0.0f;
    float dvz_dx = 0.0f, dvz_dy = 0.0f, dvz_dz = 0.0f;
    float totalWeight = 0.0f;
    int validNeighbors = 0;

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
            const glm::vec3 velDiff = glm::vec3(neighborVel - particles.velocities[id]);
            const float gradW = device::densityDerivativeKernel(dist, h);
            const float weight = particles.masses[neighbourIdx] / particles.densities[neighbourIdx];
            const glm::vec3 gradDir = r / dist;

            dvx_dx += weight * velDiff.x * gradDir.x * gradW;
            dvx_dy += weight * velDiff.x * gradDir.y * gradW;
            dvx_dz += weight * velDiff.x * gradDir.z * gradW;
            dvy_dx += weight * velDiff.y * gradDir.x * gradW;
            dvy_dy += weight * velDiff.y * gradDir.y * gradW;
            dvy_dz += weight * velDiff.y * gradDir.z * gradW;
            dvz_dx += weight * velDiff.z * gradDir.x * gradW;
            dvz_dy += weight * velDiff.z * gradDir.y * gradW;
            dvz_dz += weight * velDiff.z * gradDir.z * gradW;
            totalWeight += weight * fabsf(gradW);
            validNeighbors++;
        }
    });
    if (validNeighbors < 4)
    {
        return -1;
    }

    if (totalWeight > 0.0f)
    {
        float invWeight = 1.0f / totalWeight;
        dvx_dx *= invWeight;
        dvx_dy *= invWeight;
        dvx_dz *= invWeight;
        dvy_dx *= invWeight;
        dvy_dy *= invWeight;
        dvy_dz *= invWeight;
        dvz_dx *= invWeight;
        dvz_dy *= invWeight;
        dvz_dz *= invWeight;
    }

    float omega_x = dvz_dy - dvy_dz;
    float omega_y = dvx_dz - dvz_dx;
    float omega_z = dvy_dx - dvx_dy;

    float vorticityMagnitude = glm::length(glm::vec3(omega_x, omega_y, omega_z));
    if (vorticityMagnitude > _vorticity.split.minimalVorticityThreshold)
    {
        return vorticityMagnitude;
    }

    return -1;
}

__device__ auto MergeCriterionGenerator::operator()(ParticlesData particles,
                                                    uint32_t id,
                                                    const SphSimulation::Grid& grid,
                                                    const Simulation::Parameters& simulationData) const -> float
{
    if (particles.masses[id] > _maximalMass)
    {
        return -1;
    }

    const glm::vec4 position = particles.positions[id];
    const float h = particles.smoothingRadiuses[id];
    float dvx_dx = 0.0f, dvx_dy = 0.0f, dvx_dz = 0.0f;
    float dvy_dx = 0.0f, dvy_dy = 0.0f, dvy_dz = 0.0f;
    float dvz_dx = 0.0f, dvz_dy = 0.0f, dvz_dz = 0.0f;
    float totalWeight = 0.0f;
    int validNeighbors = 0;
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
            const glm::vec3 velDiff = glm::vec3(neighborVel - particles.velocities[id]);
            const float gradW = device::densityDerivativeKernel(dist, h);
            const float weight = particles.masses[neighbourIdx] / particles.densities[neighbourIdx];
            const glm::vec3 gradDir = r / dist;
            dvx_dx += weight * velDiff.x * gradDir.x * gradW;
            dvx_dy += weight * velDiff.x * gradDir.y * gradW;
            dvx_dz += weight * velDiff.x * gradDir.z * gradW;
            dvy_dx += weight * velDiff.y * gradDir.x * gradW;
            dvy_dy += weight * velDiff.y * gradDir.y * gradW;
            dvy_dz += weight * velDiff.y * gradDir.z * gradW;
            dvz_dx += weight * velDiff.z * gradDir.x * gradW;
            dvz_dy += weight * velDiff.z * gradDir.y * gradW;
            dvz_dz += weight * velDiff.z * gradDir.z * gradW;
            totalWeight += weight * fabsf(gradW);
            validNeighbors++;
        }
    });

    if (validNeighbors < 4)
    {
        return -1;
    }
    if (totalWeight > 0.0f)
    {
        float invWeight = 1.0f / totalWeight;
        dvx_dx *= invWeight;
        dvx_dy *= invWeight;
        dvx_dz *= invWeight;
        dvy_dx *= invWeight;
        dvy_dy *= invWeight;
        dvy_dz *= invWeight;
        dvz_dx *= invWeight;
        dvz_dy *= invWeight;
        dvz_dz *= invWeight;
    }

    float omega_x = dvz_dy - dvy_dz;
    float omega_y = dvx_dz - dvz_dx;
    float omega_z = dvy_dx - dvx_dy;
    float vorticityMagnitude = glm::length(glm::vec3(omega_x, omega_y, omega_z));

    if (vorticityMagnitude < _vorticity.merge.maximalVorticityThreshold)
    {
        return vorticityMagnitude / _vorticity.merge.maximalVorticityThreshold;
    }

    return -1;
}

}
