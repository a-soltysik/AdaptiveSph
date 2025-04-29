#define FMT_UNICODE 0

#include <cuda_runtime.h>
#include <fmt/ranges.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#include <__msvc_ranges_to.hpp>
#include <thrust/detail/scan.inl>
#include <vector>

#include "AdaptiveSphSimulation.cuh"
#include "Span.cuh"
#include "common/Utils.cuh"
#include "fmt/printf.h"
#include "glm/gtx/string_cast.hpp"
#include "refinement/Common.cuh"
#include "refinement/ParticleOperations.cuh"

namespace sph::cuda
{

AdaptiveSphSimulation::AdaptiveSphSimulation(const Parameters& initialParameters,
                                             const std::vector<glm::vec4>& positions,
                                             const ParticlesDataBuffer& memory,
                                             const refinement::RefinementParameters& refinementParams)
    : SphSimulation(initialParameters, positions, memory, refinementParams.maxParticleCount),
      _refinementParams(refinementParams),
      _refinementData {initializeRefinementData(_refinementParams.maxParticleCount, _refinementParams.maxBatchRatio)}
{
    const auto initialCount = SphSimulation::getParticlesCount();
    cudaMemcpy(_refinementData.particlesCount, &initialCount, sizeof(uint32_t), cudaMemcpyHostToDevice);
}

AdaptiveSphSimulation::~AdaptiveSphSimulation() { }

refinement::RefinementData AdaptiveSphSimulation::initializeRefinementData(uint32_t maxParticleCount,
                                                                           float maxBatchSize)
{
    uint32_t* particlesIdsToSplit = nullptr;
    uint32_t* particlesIds = nullptr;
    uint32_t* particlesSplitCount = nullptr;
    uint32_t* particlesCount = nullptr;
    float* criterionValuesSplit = nullptr;
    float* criterionValuesMerge = nullptr;

    uint32_t* particlesIdsToMergeFirst = nullptr;
    uint32_t* particlesIdsToMergeSecond = nullptr;
    uint32_t* particlesMergeCount = nullptr;
    uint32_t* removalMarks = nullptr;
    uint32_t* prefixSums = nullptr;

    cudaMalloc(&particlesIdsToSplit, maxParticleCount * maxBatchSize * sizeof(uint32_t));
    cudaMalloc(&particlesSplitCount, sizeof(uint32_t));
    cudaMalloc(&particlesCount, sizeof(uint32_t));
    cudaMalloc(&criterionValuesSplit, maxParticleCount * sizeof(float));
    cudaMalloc(&criterionValuesMerge, maxParticleCount * sizeof(float));
    cudaMalloc(&particlesIds, maxParticleCount * sizeof(uint32_t));

    cudaMalloc(&particlesIdsToMergeFirst, maxParticleCount * maxBatchSize * sizeof(uint32_t));
    cudaMalloc(&particlesIdsToMergeSecond, maxParticleCount * maxBatchSize * sizeof(uint32_t));
    cudaMalloc(&particlesMergeCount, sizeof(uint32_t));
    cudaMalloc(&removalMarks, maxParticleCount * sizeof(uint32_t));
    cudaMalloc(&prefixSums, maxParticleCount * sizeof(uint32_t));

    return {
        .split = {.criterionValues = {.data = criterionValuesSplit, .size = maxParticleCount},
                  .particlesIdsToSplit = {.data = particlesIdsToSplit,
                                          .size = static_cast<size_t>(maxParticleCount * maxBatchSize)},
                  .particlesSplitCount = particlesSplitCount},
        .merge = {.criterionValues = {.data = criterionValuesMerge, .size = maxParticleCount},
                  .particlesIdsToMerge = {Span {.data = particlesIdsToMergeFirst,
                                                .size = static_cast<size_t>(maxParticleCount * maxBatchSize)},
                                          Span {.data = particlesIdsToMergeSecond,
                                                .size = static_cast<size_t>(maxParticleCount * maxBatchSize)}},
                  .removalFlags = {.data = removalMarks, .size = maxParticleCount},
                  .prefixSums = {.data = prefixSums, .size = maxParticleCount},
                  .particlesMergeCount = particlesMergeCount},
        .particlesIds = {.data = particlesIds, .size = maxParticleCount},
        .particlesCount = particlesCount
    };
}

void AdaptiveSphSimulation::update(const Parameters& parameters, float deltaTime)
{
    // First perform standard SPH steps
    // But use variable smoothing length versions of density and pressure computation
    updateParameters(parameters);

    computeExternalForces(deltaTime);
    resetGrid();
    assignParticlesToCells();
    sortParticles();
    calculateCellStartAndEndIndices();
    computeDensities();
    computePressureForce(deltaTime);

    // After refinement starts, use variable smoothing length methods
    //refinement::computeDensitiesWithVariableSmoothingLengths<<<SphSimulation::getBlocksPerGridForParticles(),
    //                                                           getThreadsPerBlock()>>>(getParticles(),
    //                                                                                   getState(),
    //                                                                                   getParameters());
    //
    //refinement::computePressureForceWithVariableSmoothingLengths<<<SphSimulation::getBlocksPerGridForParticles(),
    //                                                               getThreadsPerBlock()>>>(getParticles(),
    //                                                                                       getState(),
    //                                                                                       getParameters(),
    //                                                                                       deltaTime);

    computeViscosityForce(deltaTime);
    integrateMotion(deltaTime);
    handleCollisions();

    if (_frameCounter == _refinementParams.initialCooldown ||
        (_frameCounter > _refinementParams.initialCooldown && _frameCounter % _refinementParams.cooldown == 0))
    {
        performAdaptiveRefinement(deltaTime);
    }

    cudaDeviceSynchronize();
    _frameCounter++;
}

void AdaptiveSphSimulation::updateParticleCount()
{
    setParticleCount(fromGpu(_refinementData.particlesCount));
}

auto AdaptiveSphSimulation::getBlocksPerGridForParticles(uint32_t count) const -> dim3
{
    return {(count + getThreadsPerBlock() - 1) / getThreadsPerBlock()};
}

void AdaptiveSphSimulation::performAdaptiveRefinement(float deltaTime)
{
    resetRefinementCounters();
    identifyAndSplitParticles();
    identifyAndMergeParticles();
    updateParticleCount();
}

void AdaptiveSphSimulation::resetRefinementCounters() const
{
    uint32_t zero = 0;
    cudaMemcpy(_refinementData.split.particlesSplitCount, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(_refinementData.merge.particlesMergeCount, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
}

void AdaptiveSphSimulation::identifyAndSplitParticles() const
{
    refinement::getCriterionValues<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
        getParticles(),
        _refinementData.split.criterionValues,
        [minMassRatio = _refinementParams.minMassRatio,
         initialMass = getInitialMass()] __device__(ParticlesData particles, const auto idx) {
            if (particles.masses[idx] < minMassRatio * initialMass)
            {
                return FLT_MAX;
            }
            const auto velocity = particles.velocities[idx];
            const auto velocityMagnitude = glm::length(glm::vec3(velocity));
            if (velocityMagnitude < 2.5F)
            {
                return FLT_MAX;
            }
            return velocityMagnitude;
        });

    refinement::findTopParticlesToSplit(getParticles(), _refinementData, _refinementParams, thrust::greater<float> {});

    const auto particlesToSplitCount = fromGpu(_refinementData.split.particlesSplitCount);

    if (particlesToSplitCount == 0)
    {
        return;
    }

    refinement::splitParticles<<<getBlocksPerGridForParticles(particlesToSplitCount), getThreadsPerBlock()>>>(
        getParticles(),
        _refinementData,
        _refinementParams.splitting,
        _refinementParams.maxParticleCount);
}

void AdaptiveSphSimulation::identifyAndMergeParticles() const
{
    if (getParticlesCount() <= 1)
    {
        return;
    }

    thrust::fill(thrust::device,
                 _refinementData.merge.removalFlags.data,
                 _refinementData.merge.removalFlags.data + getParticlesCount(),
                 0);

    refinement::getCriterionValues<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
        getParticles(),
        _refinementData.merge.criterionValues,
        [maxMassRatio = _refinementParams.maxMassRatio,
         initialMass = getInitialMass()] __device__(ParticlesData particles, const auto idx) {
            if (particles.masses[idx] > maxMassRatio * initialMass)
            {
                return FLT_MAX;
            }
            const auto velocity = particles.velocities[idx];
            const auto velocityMagnitude = glm::length(glm::vec3(velocity));
            if (velocityMagnitude > 0.5F)
            {
                return FLT_MAX;
            }
            return velocityMagnitude;
        });

    refinement::findTopParticlesToMerge(getParticles(), _refinementData, _refinementParams, thrust::less<float> {});

    const auto particlesToMergeCount = fromGpu(_refinementData.merge.particlesMergeCount);

    if (particlesToMergeCount == 0)
    {
        return;
    }

    const uint32_t maxProcessCount = std::min(
        particlesToMergeCount,
        static_cast<uint32_t>(getParticlesCount() * _refinementParams.maxBatchRatio));  // Only merge up to 10% at once
    cudaMemcpy(_refinementData.merge.particlesMergeCount, &maxProcessCount, sizeof(uint32_t), cudaMemcpyHostToDevice);

    refinement::getMergeCandidates<<<getBlocksPerGridForParticles(maxProcessCount), getThreadsPerBlock()>>>(
        getParticles(),
        _refinementData,
        getState().grid,
        getParameters());

    // Phase 1: Mark particles for merging
    refinement::markPotentialMerges<<<getBlocksPerGridForParticles(maxProcessCount), getThreadsPerBlock()>>>(
        getParticles(),
        _refinementData);

    // Phase 2: Perform actual merging only on marked pairs
    refinement::mergeParticles<<<getBlocksPerGridForParticles(maxProcessCount), getThreadsPerBlock()>>>(
        getParticles(),
        _refinementData);

    computePrefixSum();

    refinement::updateParticleCount<<<1, 1>>>(_refinementData, getParticlesCount());

    refinement::removeParticles<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
        getParticles(),
        _refinementData);
}

__global__ void validateRemovalFlags(ParticlesData particles, refinement::RefinementData refinementData)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }

    // Ensure the removal flag is valid (0, 1, or 2)
    const auto flag = refinementData.merge.removalFlags.data[idx];
    if (flag > 2)
    {
        // Reset invalid flags to 0 (keep the particle)
        refinementData.merge.removalFlags.data[idx] = 0;
    }
}

void AdaptiveSphSimulation::computePrefixSum() const
{
    validateRemovalFlags<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(getParticles(),
                                                                                                  _refinementData);

    // Create a temporary array for the scan, containing only 0 or 1 (1 for particles to remove)
    thrust::transform(thrust::device,
                      _refinementData.merge.removalFlags.data,
                      _refinementData.merge.removalFlags.data + getParticlesCount(),
                      _refinementData.merge.prefixSums.data,
                      [] __device__(uint32_t flag) {
                          // Only count particles marked for removal (value 2)
                          return (flag == 2) ? 1 : 0;
                      });

    thrust::exclusive_scan(thrust::device,
                           _refinementData.merge.prefixSums.data,
                           _refinementData.merge.prefixSums.data + getParticlesCount(),
                           _refinementData.merge.prefixSums.data,
                           0);
}

}  // namespace sph::cuda
