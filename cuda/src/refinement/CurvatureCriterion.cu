// CurvatureCriterion.cu
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
        return FLT_MAX;
    }
    const glm::vec4 position = particles.positions[id];
    const float h = particles.smoothingRadiuses[id];
    const float density = particles.densities[id];
    // Calculate density Laplacian (measure of curvature)
    float densityLaplacian = 0.0f;
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
            const float neighborDensity = particles.densities[neighbourIdx];
            const float densityDiff = neighborDensity - density;
            // Use viscosity Laplacian kernel
            const float lapW = device::viscosityLaplacianKernel(dist, h);
            const float weight = particles.masses[neighbourIdx] / particles.densities[neighbourIdx];
            densityLaplacian += weight * densityDiff * lapW;
            validNeighbors++;
        }
    });
    if (validNeighbors < 3)
    {
        return FLT_MAX;  // Not enough neighbors
    }
    float curvatureMagnitude = fabsf(densityLaplacian);
    if (curvatureMagnitude > _curvature.threshold)
    {
        return curvatureMagnitude * _curvature.scaleFactor;
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

    // Calculate curvature to check if it's low enough for merging
    SplitCriterionGenerator splitGen(0.0f, _curvature);
    float splitValue = splitGen(particles, id, grid, simulationData);

    // Use lower threshold for merging
    if (splitValue != FLT_MAX && splitValue > _curvature.threshold * 0.5f)
    {
        return FLT_MAX;  // High curvature, don't merge
    }

    return splitValue == FLT_MAX ? 0.0f : splitValue;
}

}
