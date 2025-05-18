#include <cstdint>
#include <glm/common.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/geometric.hpp>

#include "CurvatureCriterion.cuh"
#include "algorithm/kernels/Kernel.cuh"
#include "cuda/Simulation.cuh"
#include "simulation/adaptive/SphSimulation.cuh"
#include "utils/Iteration.cuh"

namespace sph::cuda::refinement::curvature
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
    const auto position = particles.positions[id];
    const auto smoothingRadius = particles.smoothingRadiuses[id];
    const auto density = particles.densities[id];

    auto densityLaplacian = 0.F;
    auto totalWeight = 0.F;
    auto validNeighbors = uint32_t {};

    forEachNeighbour(position,
                     particles,
                     simulationData,
                     grid,
                     [&](const auto neighbourIdx, const glm::vec4& adjustedPos) {
                         if (neighbourIdx == id)
                         {
                             return;
                         }

                         const auto r = glm::vec3(adjustedPos - position);
                         const auto dist = glm::length(r);
                         if (dist < smoothingRadius)
                         {
                             const auto neighborDensity = particles.densities[neighbourIdx];
                             const auto densityDiff = neighborDensity - density;
                             const auto lapW = device::densityLaplacianKernel(dist, smoothingRadius);
                             const auto weight = particles.masses[neighbourIdx] / particles.densities[neighbourIdx];
                             densityLaplacian += weight * densityDiff * lapW;
                             totalWeight += weight * lapW;
                             validNeighbors++;
                         }
                     });
    if (validNeighbors < 3)
    {
        return -1;
    }

    if (totalWeight > 0.0F)
    {
        densityLaplacian /= totalWeight;
    }

    const auto curvatureMagnitude = glm::abs(densityLaplacian);
    if (curvatureMagnitude > _curvature.split.minimalCurvatureThreshold)
    {
        return curvatureMagnitude;
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

    const auto position = particles.positions[id];
    const auto smoothingRadius = particles.smoothingRadiuses[id];
    const auto density = particles.densities[id];

    auto densityLaplacian = 0.F;
    auto totalWeight = 0.F;
    auto validNeighbors = uint32_t {};
    forEachNeighbour(position,
                     particles,
                     simulationData,
                     grid,
                     [&](const auto neighbourIdx, const glm::vec4& adjustedPos) {
                         if (neighbourIdx == id)
                         {
                             return;
                         }

                         const auto r = glm::vec3(adjustedPos - position);
                         const auto dist = glm::length(r);
                         if (dist < smoothingRadius)
                         {
                             const auto neighborDensity = particles.densities[neighbourIdx];
                             const auto densityDiff = neighborDensity - density;

                             const auto lapW = device::densityLaplacianKernel(dist, smoothingRadius);
                             const auto weight = particles.masses[neighbourIdx] / particles.densities[neighbourIdx];
                             densityLaplacian += weight * densityDiff * lapW;
                             totalWeight += weight * glm::abs(lapW);
                             validNeighbors++;
                         }
                     });

    if (validNeighbors < 3)
    {
        return -1;
    }

    if (totalWeight > 0.0F)
    {
        densityLaplacian /= totalWeight;
    }

    const auto curvatureMagnitude = glm::abs(densityLaplacian);

    if (curvatureMagnitude < _curvature.merge.maximalCurvatureThreshold)
    {
        return curvatureMagnitude / _curvature.merge.maximalCurvatureThreshold;
    }

    return -1;
}

}
