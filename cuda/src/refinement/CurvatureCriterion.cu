#include <cfloat>
#include <glm/geometric.hpp>

#include "../common/Iteration.hpp"
#include "../common/Utils.cuh"
#include "../device/Kernel.cuh"
#include "CurvatureCriterion.cuh"

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
    const glm::vec4 position = particles.positions[id];
    const float h = particles.smoothingRadiuses[id];
    const float density = particles.densities[id];

    float densityLaplacian = 0.0f;
    float totalWeight = 0.0f;
    int validNeighbors = 0;

    forEachNeighbour(position,
                     particles,
                     simulationData,
                     grid,
                     [&](const auto neighbourIdx, const glm::vec4& adjustedPos) {
                         if (neighbourIdx == id)
                         {
                             return;
                         }

                         const glm::vec3 r = glm::vec3(adjustedPos - position);
                         const float dist = glm::length(r);
                         if (dist < h && dist > 1e-5f)
                         {
                             const float neighborDensity = particles.densities[neighbourIdx];
                             const float densityDiff = neighborDensity - density;
                             // Use proper Laplacian kernel
                             const float lapW = device::wendlandLaplacianKernel(dist, h);
                             const float weight = particles.masses[neighbourIdx] / particles.densities[neighbourIdx];
                             densityLaplacian += weight * densityDiff * lapW;
                             totalWeight += weight * lapW;
                             validNeighbors++;
                         }
                     });
    if (validNeighbors < 3)
    {
        return -1;
    }

    if (totalWeight > 0.0f)
    {
        densityLaplacian /= totalWeight;
    }

    float curvatureMagnitude = fabsf(densityLaplacian);
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

    const glm::vec4 position = particles.positions[id];
    const float h = particles.smoothingRadiuses[id];
    const float density = particles.densities[id];

    float densityLaplacian = 0.0f;
    float totalWeight = 0.0f;
    int validNeighbors = 0;
    forEachNeighbour(position,
                     particles,
                     simulationData,
                     grid,
                     [&](const auto neighbourIdx, const glm::vec4& adjustedPos) {
                         if (neighbourIdx == id)
                         {
                             return;
                         }

                         const glm::vec3 r = glm::vec3(adjustedPos - position);
                         const float dist = glm::length(r);
                         if (dist < h && dist > 1e-5f)
                         {
                             const float neighborDensity = particles.densities[neighbourIdx];
                             const float densityDiff = neighborDensity - density;

                             const float lapW = device::wendlandLaplacianKernel(dist, h);
                             const float weight = particles.masses[neighbourIdx] / particles.densities[neighbourIdx];
                             densityLaplacian += weight * densityDiff * lapW;
                             totalWeight += weight * fabsf(lapW);
                             validNeighbors++;
                         }
                     });

    if (validNeighbors < 3)
    {
        return -1;
    }

    if (totalWeight > 0.0f)
    {
        densityLaplacian /= totalWeight;
    }

    float curvatureMagnitude = fabsf(densityLaplacian);

    if (curvatureMagnitude < _curvature.merge.maximalCurvatureThreshold)
    {
        return curvatureMagnitude / _curvature.merge.maximalCurvatureThreshold;
    }

    return -1;
}

}
