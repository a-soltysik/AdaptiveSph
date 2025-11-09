#pragma once
#include <cstdint>

#include "algorithm/kernels/Kernel.cuh"
#include "cuda/Simulation.cuh"
#include "cuda/refinement/RefinementParameters.cuh"
#include "simulation/SphSimulation.cuh"
#include "utils/Iteration.cuh"

namespace sph::cuda::refinement::vorticity
{

class SplitCriterionGenerator
{
public:
    __host__ __device__ SplitCriterionGenerator(float minimalMass, VorticityParameters vorticity)
        : _minimalMass(minimalMass),
          _vorticity(vorticity)
    {
    }

    __device__ auto operator()(ParticlesData particles,
                               uint32_t id,
                               const SphSimulation::Grid& grid,
                               const Simulation::Parameters& simulationData) const -> float;

private:
    float _minimalMass;
    VorticityParameters _vorticity;
};

class MergeCriterionGenerator
{
public:
    __host__ __device__ MergeCriterionGenerator(float maximalMass, VorticityParameters vorticity)
        : _maximalMass(maximalMass),
          _vorticity(vorticity)
    {
    }

    __device__ auto operator()(ParticlesData particles,
                               uint32_t id,
                               const SphSimulation::Grid& grid,
                               const Simulation::Parameters& simulationData) const -> float;

private:
    float _maximalMass;
    VorticityParameters _vorticity;
};

inline __device__ auto SplitCriterionGenerator::operator()(ParticlesData particles,
                                                           uint32_t id,
                                                           const SphSimulation::Grid& grid,
                                                           const Simulation::Parameters& simulationData) const -> float
{
    if (particles.masses[id] < _minimalMass)
    {
        return -1;
    }

    const auto position = particles.positions[id];
    const auto smoothingRadius = particles.smoothingRadiuses[id];

    auto dvx_dx = 0.0F;
    auto dvx_dy = 0.0F;
    auto dvx_dz = 0.0F;
    auto dvy_dx = 0.0F;
    auto dvy_dy = 0.0F;
    auto dvy_dz = 0.0F;
    auto dvz_dx = 0.0F;
    auto dvz_dy = 0.0F;
    auto dvz_dz = 0.0F;
    auto totalWeight = 0.0F;
    auto validNeighbors = 0;

    forEachNeighbour(position,
                     particles,
                     simulationData,
                     grid,
                     smoothingRadius,
                     [&](const auto neighbourIdx, const glm::vec4& adjustedPos) {
                         if (neighbourIdx == id)
                         {
                             return;
                         }

                         const auto r = glm::vec3(adjustedPos - position);
                         const auto dist = glm::length(r);
                         if (dist < smoothingRadius && dist > 1e-5f)
                         {
                             const auto neighborVel = particles.velocities[neighbourIdx];
                             const auto velDiff = glm::vec3(neighborVel - particles.velocities[id]);
                             const auto gradW = device::densityDerivativeKernel(dist, smoothingRadius);
                             const auto weight = particles.masses[neighbourIdx] / particles.densities[neighbourIdx];
                             const auto gradDir = r / dist;

                             dvx_dx += weight * velDiff.x * gradDir.x * gradW;
                             dvx_dy += weight * velDiff.x * gradDir.y * gradW;
                             dvx_dz += weight * velDiff.x * gradDir.z * gradW;
                             dvy_dx += weight * velDiff.y * gradDir.x * gradW;
                             dvy_dy += weight * velDiff.y * gradDir.y * gradW;
                             dvy_dz += weight * velDiff.y * gradDir.z * gradW;
                             dvz_dx += weight * velDiff.z * gradDir.x * gradW;
                             dvz_dy += weight * velDiff.z * gradDir.y * gradW;
                             dvz_dz += weight * velDiff.z * gradDir.z * gradW;
                             totalWeight += weight * glm::abs(gradW);
                             validNeighbors++;
                         }
                     });
    if (validNeighbors < 4)
    {
        return -1;
    }

    if (totalWeight > 0.0F)
    {
        const auto invWeight = 1.0F / totalWeight;
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

    const auto omega_x = dvz_dy - dvy_dz;
    const auto omega_y = dvx_dz - dvz_dx;
    const auto omega_z = dvy_dx - dvx_dy;

    const auto vorticityMagnitude = glm::length(glm::vec3(omega_x, omega_y, omega_z));
    if (vorticityMagnitude > _vorticity.split.minimalVorticityThreshold)
    {
        return vorticityMagnitude;
    }

    return -1;
}

inline __device__ auto MergeCriterionGenerator::operator()(ParticlesData particles,
                                                           uint32_t id,
                                                           const SphSimulation::Grid& grid,
                                                           const Simulation::Parameters& simulationData) const -> float
{
    if (particles.masses[id] > _maximalMass)
    {
        return -1;
    }

    const auto position = particles.positions[id];
    const auto smoothingRadius = particles.smoothingRadiuses[id];
    auto dvx_dx = 0.0F, dvx_dy = 0.0F, dvx_dz = 0.0F;
    auto dvy_dx = 0.0F, dvy_dy = 0.0F, dvy_dz = 0.0F;
    auto dvz_dx = 0.0F, dvz_dy = 0.0F, dvz_dz = 0.0F;
    auto totalWeight = 0.0F;
    auto validNeighbors = uint32_t {};
    forEachNeighbour(position,
                     particles,
                     simulationData,
                     grid,
                     smoothingRadius,
                     [&](const auto neighbourIdx, const glm::vec4& adjustedPos) {
                         if (neighbourIdx == id)
                         {
                             return;
                         }

                         const auto r = glm::vec3(adjustedPos - position);
                         const auto dist = glm::length(r);
                         if (dist < smoothingRadius && dist > 1e-5f)
                         {
                             const auto neighborVel = particles.velocities[neighbourIdx];
                             const auto velDiff = glm::vec3(neighborVel - particles.velocities[id]);
                             const auto gradW = device::densityDerivativeKernel(dist, smoothingRadius);
                             const auto weight = particles.masses[neighbourIdx] / particles.densities[neighbourIdx];
                             const auto gradDir = r / dist;
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
    if (totalWeight > 0.0F)
    {
        float invWeight = 1.0F / totalWeight;
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

    const auto omega_x = dvz_dy - dvy_dz;
    const auto omega_y = dvx_dz - dvz_dx;
    const auto omega_z = dvy_dx - dvx_dy;
    const auto vorticityMagnitude = glm::length(glm::vec3(omega_x, omega_y, omega_z));

    if (vorticityMagnitude < _vorticity.merge.maximalVorticityThreshold)
    {
        return vorticityMagnitude / _vorticity.merge.maximalVorticityThreshold;
    }

    return -1;
}

}
