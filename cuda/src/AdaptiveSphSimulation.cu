#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <vector_types.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "AdaptiveSphSimulation.cuh"
#include "Span.cuh"
#include "SphSimulation.cuh"
#include "common/Utils.cuh"
#include "cuda/Simulation.cuh"
#include "cuda/refinement/RefinementParameters.cuh"
#include "refinement/Common.cuh"
#include "refinement/ParticleOperations.cuh"
#include "refinement/VelocityCriterion.cuh"

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

AdaptiveSphSimulation::~AdaptiveSphSimulation()
{
    cudaFree(_refinementData.split.criterionValues.data);
    cudaFree(_refinementData.split.particlesIdsToSplit.data);
    cudaFree(_refinementData.split.particlesSplitCount);

    // Free merge-related memory
    cudaFree(_refinementData.merge.criterionValues.data);
    cudaFree(_refinementData.merge.particlesIdsToMerge.first.data);
    cudaFree(_refinementData.merge.particlesIdsToMerge.second.data);
    cudaFree(_refinementData.merge.removalFlags.data);
    cudaFree(_refinementData.merge.prefixSums.data);
    cudaFree(_refinementData.merge.particlesMergeCount);

    // Free shared memory
    cudaFree(_refinementData.particlesIds.data);
    cudaFree(_refinementData.particlesCount);
}

auto AdaptiveSphSimulation::initializeRefinementData(uint32_t maxParticleCount, float maxBatchSize)
    -> refinement::RefinementData
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
    refinement::RefinementData::RemovalState* removalMarks = nullptr;
    uint32_t* prefixSums = nullptr;

    cudaMalloc(&particlesIdsToSplit,
               static_cast<size_t>(static_cast<float>(maxParticleCount) * maxBatchSize) * sizeof(uint32_t));
    cudaMalloc(&particlesSplitCount, sizeof(uint32_t));
    cudaMalloc(&particlesCount, sizeof(uint32_t));
    cudaMalloc(&criterionValuesSplit, maxParticleCount * sizeof(float));
    cudaMalloc(&criterionValuesMerge, maxParticleCount * sizeof(float));
    cudaMalloc(&particlesIds, maxParticleCount * sizeof(uint32_t));

    cudaMalloc(&particlesIdsToMergeFirst,
               static_cast<size_t>(static_cast<float>(maxParticleCount) * maxBatchSize) * sizeof(uint32_t));
    cudaMalloc(&particlesIdsToMergeSecond,
               static_cast<size_t>(static_cast<float>(maxParticleCount) * maxBatchSize) * sizeof(uint32_t));
    cudaMalloc(&particlesMergeCount, sizeof(uint32_t));
    cudaMalloc(&removalMarks, maxParticleCount * sizeof(uint32_t));
    cudaMalloc(&prefixSums, maxParticleCount * sizeof(uint32_t));

    return {
        .split = {.criterionValues = {.data = criterionValuesSplit, .size = maxParticleCount},
                  .particlesIdsToSplit = {.data = particlesIdsToSplit,
                                          .size =
                                              static_cast<size_t>(static_cast<float>(maxParticleCount) * maxBatchSize)},
                  .particlesSplitCount = particlesSplitCount},
        .merge = {.criterionValues = {.data = criterionValuesMerge, .size = maxParticleCount},
                  .particlesIdsToMerge =
                      {Span {.data = particlesIdsToMergeFirst,
                             .size = static_cast<size_t>(static_cast<float>(maxParticleCount) * maxBatchSize)},
                       Span {.data = particlesIdsToMergeSecond,
                             .size = static_cast<size_t>(static_cast<float>(maxParticleCount) * maxBatchSize)}},
                  .removalFlags = {.data = removalMarks, .size = maxParticleCount},
                  .prefixSums = {.data = prefixSums, .size = maxParticleCount},
                  .particlesMergeCount = particlesMergeCount},
        .particlesIds = {.data = particlesIds, .size = maxParticleCount},
        .particlesCount = particlesCount
    };
}

void AdaptiveSphSimulation::update(const Parameters& parameters, float deltaTime)
{
    updateParameters(parameters);

    computeExternalForces(deltaTime);
    resetGrid();
    assignParticlesToCells();
    sortParticles();
    calculateCellStartAndEndIndices();
    computeDensities();
    computePressureForce(deltaTime);
    computeViscosityForce(deltaTime);
    integrateMotion(deltaTime);
    handleCollisions();

    if (_frameCounter == _refinementParams.initialCooldown ||
        (_frameCounter > _refinementParams.initialCooldown && _frameCounter % _refinementParams.cooldown == 0))
    {
        performAdaptiveRefinement();
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

void AdaptiveSphSimulation::performAdaptiveRefinement()
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
        refinement::velocity::SplitCriterionGenerator {_refinementParams.minMassRatio * getInitialMass(), 2.5F});

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
                 refinement::RefinementData::RemovalState::Default);

    refinement::getCriterionValues<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
        getParticles(),
        _refinementData.merge.criterionValues,
        refinement::velocity::MergeCriterionGenerator {_refinementParams.maxMassRatio * getInitialMass(), 0.5F});

    refinement::findTopParticlesToMerge(getParticles(), _refinementData, _refinementParams, thrust::less<float> {});

    const auto particlesToMergeCount = fromGpu(_refinementData.merge.particlesMergeCount);

    if (particlesToMergeCount == 0)
    {
        return;
    }

    refinement::getMergeCandidates<<<getBlocksPerGridForParticles(particlesToMergeCount), getThreadsPerBlock()>>>(
        getParticles(),
        _refinementData,
        getState().grid,
        getParameters());

    refinement::markPotentialMerges<<<getBlocksPerGridForParticles(particlesToMergeCount), getThreadsPerBlock()>>>(
        getParticles(),
        _refinementData);

    refinement::mergeParticles<<<getBlocksPerGridForParticles(particlesToMergeCount), getThreadsPerBlock()>>>(
        getParticles(),
        _refinementData);

    computePrefixSum();

    refinement::updateParticleCount<<<1, 1>>>(_refinementData, getParticlesCount());

    refinement::removeParticles<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
        getParticles(),
        _refinementData);
}

void AdaptiveSphSimulation::computePrefixSum() const
{
    thrust::transform(thrust::device,
                      _refinementData.merge.removalFlags.data,
                      _refinementData.merge.removalFlags.data + getParticlesCount(),
                      _refinementData.merge.prefixSums.data,
                      [] __device__(refinement::RefinementData::RemovalState flag) {
                          return (flag == refinement::RefinementData::RemovalState::Remove) ? 1 : 0;
                      });

    thrust::exclusive_scan(thrust::device,
                           _refinementData.merge.prefixSums.data,
                           _refinementData.merge.prefixSums.data + getParticlesCount(),
                           _refinementData.merge.prefixSums.data,
                           0);
}

}
