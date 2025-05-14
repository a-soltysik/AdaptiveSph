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
#include "refinement/CurvatureCriterion.cuh"
#include "refinement/InterfaceCriterion.cuh"
#include "refinement/ParticleOperations.cuh"
#include "refinement/VelocityCriterion.cuh"
#include "refinement/VorticityCriterion.cuh"

namespace sph::cuda
{

template <typename CriterionGenerator>
__global__ void getCriterionValuesWithGrid(ParticlesData particles,
                                           Span<float> splitCriterionValues,
                                           CriterionGenerator criterionGenerator,
                                           const SphSimulation::Grid grid,
                                           const Simulation::Parameters simulationData)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }
    const auto value = criterionGenerator(particles, idx, grid, simulationData);
    splitCriterionValues.data[idx] = value;
}

template <typename CriterionGenerator>
__global__ void getCriterionValuesNoGrid(ParticlesData particles,
                                         Span<float> splitCriterionValues,
                                         CriterionGenerator criterionGenerator,
                                         const Simulation::Parameters simulationData)
{
    const auto idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= particles.particleCount)
    {
        return;
    }
    const auto value = criterionGenerator(particles, idx, simulationData);
    splitCriterionValues.data[idx] = value;
}

AdaptiveSphSimulation::AdaptiveSphSimulation(const Parameters& initialParameters,
                                             const std::vector<glm::vec4>& positions,
                                             const ParticlesDataBuffer& memory,
                                             const refinement::RefinementParameters& refinementParams)
    : SphSimulation(initialParameters, positions, memory, refinementParams.maxParticleCount),
      _refinementParams(refinementParams),
      _refinementData {initializeRefinementData(_refinementParams.maxParticleCount, _refinementParams.maxBatchRatio)},
      _enhancedMergeData {allocateEnhancedMergeData(refinementParams.maxParticleCount)},
      _targetParticleCount {static_cast<uint32_t>(positions.size())}
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

    freeEnhancedMergeData(_enhancedMergeData);
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

    if (_frameCounter == _refinementParams.initialCooldown ||
        (_frameCounter > _refinementParams.initialCooldown && _frameCounter % _refinementParams.cooldown == 0))
    {
        performAdaptiveRefinement();
    }

    handleCollisions();

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
    uint32_t currentCount = getParticlesCount();

    resetEnhancedMergeData(currentCount);
    // First perform merging and track how many particles were removed
    performMerging();
    // Calculate the new count after merging
    uint32_t postMergeCount = getParticlesCount();
    // Calculate how many particles were removed during merging
    _particlesRemovedInLastMerge = currentCount > postMergeCount ? currentCount - postMergeCount : 0;

    // Only split particles if we've removed some during merging
    if (_particlesRemovedInLastMerge > 0)
    {
        identifyAndSplitParticles();
    }

    // Update the final particle count
    setParticleCount(fromGpu(_refinementData.particlesCount));
}

void AdaptiveSphSimulation::resetRefinementCounters() const
{
    static constexpr uint32_t zero = 0;

    cudaMemcpy(_refinementData.split.particlesSplitCount, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(_refinementData.merge.particlesMergeCount, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
    // Reset criterion values arrays
    cudaMemset(_refinementData.split.criterionValues.data,
               0,
               _refinementData.split.criterionValues.size * sizeof(float));
    cudaMemset(_refinementData.merge.criterionValues.data,
               0,
               _refinementData.merge.criterionValues.size * sizeof(float));
    // Reset particle IDs array
    cudaMemset(_refinementData.particlesIds.data, 0, _refinementData.particlesIds.size * sizeof(uint32_t));

    // Reset all removal flags and markers

    cudaMemset(_refinementData.merge.removalFlags.data, 0, _refinementData.merge.removalFlags.size * sizeof(uint32_t));
    cudaMemset(_refinementData.merge.prefixSums.data, 0, _refinementData.merge.prefixSums.size * sizeof(uint32_t));

    // Reset split particle IDs
    cudaMemset(_refinementData.split.particlesIdsToSplit.data,
               0,
               _refinementData.split.particlesIdsToSplit.size * sizeof(uint32_t));
    // Reset merge particle IDs
    cudaMemset(_refinementData.merge.particlesIdsToMerge.first.data,
               0,
               _refinementData.merge.particlesIdsToMerge.first.size * sizeof(uint32_t));
    cudaMemset(_refinementData.merge.particlesIdsToMerge.second.data,
               0,
               _refinementData.merge.particlesIdsToMerge.second.size * sizeof(uint32_t));
}

void AdaptiveSphSimulation::identifyAndSplitParticles() const
{
    const float minMass = _refinementParams.minMassRatio * getParameters().baseParticleMass;

    if (_refinementParams.criterionType == "interface")
    {
        getCriterionValuesNoGrid<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
            getParticles(),
            _refinementData.split.criterionValues,
            refinement::interfaceCriterion::SplitCriterionGenerator(minMass, _refinementParams.interfaceParameters),
            getParameters());
    }
    else if (_refinementParams.criterionType == "vorticity")
    {
        getCriterionValuesWithGrid<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
            getParticles(),
            _refinementData.split.criterionValues,
            refinement::vorticity::SplitCriterionGenerator(minMass, _refinementParams.vorticity),
            getState().grid,
            getParameters());
    }
    else if (_refinementParams.criterionType == "curvature")
    {
        getCriterionValuesWithGrid<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
            getParticles(),
            _refinementData.split.criterionValues,
            refinement::curvature::SplitCriterionGenerator(minMass, _refinementParams.curvature),
            getState().grid,
            getParameters());
    }
    else
    {
        refinement::getCriterionValues<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
            getParticles(),
            _refinementData.split.criterionValues,
            refinement::velocity::SplitCriterionGenerator(minMass, _refinementParams.velocity.split));
    }

    refinement::findTopParticlesToSplit(getParticles(), _refinementData, _refinementParams, thrust::greater<float> {});

    const auto particlesToSplitCount =
        std::min(fromGpu(_refinementData.split.particlesSplitCount), _particlesRemovedInLastMerge / 12);

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

    uint32_t currentCount = fromGpu(_refinementData.particlesCount);

    thrust::fill(thrust::device,
                 _refinementData.merge.removalFlags.data,
                 _refinementData.merge.removalFlags.data + static_cast<size_t>(currentCount),
                 refinement::RefinementData::RemovalState::Default);

    const float maxMass = _refinementParams.maxMassRatio * getParameters().baseParticleMass;

    if (_refinementParams.criterionType == "interface")
    {
        getCriterionValuesNoGrid<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
            getParticles(),
            _refinementData.merge.criterionValues,
            refinement::interfaceCriterion::MergeCriterionGenerator(maxMass, _refinementParams.interfaceParameters),
            getParameters());
    }
    else if (_refinementParams.criterionType == "vorticity")
    {
        getCriterionValuesWithGrid<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
            getParticles(),
            _refinementData.merge.criterionValues,
            refinement::vorticity::MergeCriterionGenerator(maxMass, _refinementParams.vorticity),
            getState().grid,
            getParameters());
    }
    else if (_refinementParams.criterionType == "curvature")
    {
        getCriterionValuesWithGrid<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
            getParticles(),
            _refinementData.merge.criterionValues,
            refinement::curvature::MergeCriterionGenerator(maxMass, _refinementParams.curvature),
            getState().grid,
            getParameters());
    }
    else
    {
        refinement::getCriterionValues<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
            getParticles(),
            _refinementData.merge.criterionValues,
            refinement::velocity::MergeCriterionGenerator(maxMass, _refinementParams.velocity.merge));
    }

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

    refinement::validateMergePairs<<<getBlocksPerGridForParticles(particlesToMergeCount), getThreadsPerBlock()>>>(
        _refinementData,
        getParticlesCount());

    refinement::mergeParticles<<<getBlocksPerGridForParticles(particlesToMergeCount), getThreadsPerBlock()>>>(
        getParticles(),
        _refinementData,
        getParameters());

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

void AdaptiveSphSimulation::performMerging()
{
    refinement::MergeConfiguration mergeConfig;
    mergeConfig.maxMassRatio = _refinementParams.maxMassRatio;
    mergeConfig.baseParticleMass = getParameters().baseParticleMass;
    mergeConfig.maxMassThreshold = mergeConfig.maxMassRatio * mergeConfig.baseParticleMass;

    const auto currentCount = getParticlesCount();

    cudaMemset(_enhancedMergeData.eligibleCount, 0, sizeof(uint32_t));
    cudaMemset(_enhancedMergeData.pairCount, 0, sizeof(uint32_t));

    cudaMemset(_enhancedMergeData.states.data, 0, _enhancedMergeData.states.size * sizeof(refinement::MergeState));
    cudaMemset(_enhancedMergeData.compactionMap.data, 0, _enhancedMergeData.compactionMap.size * sizeof(uint32_t));

    calculateMergeCriteria(_enhancedMergeData.criterionValues);

    refinement::identifyEligibleParticles<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
        getParticles(),
        _enhancedMergeData,
        _refinementParams.maxMassRatio * getParameters().baseParticleMass);
    cudaDeviceSynchronize();
    const uint32_t eligibleCount = fromGpu(_enhancedMergeData.eligibleCount);
    if (eligibleCount == 0)
    {
        return;
    }
    proposePartners<<<getBlocksPerGridForParticles(eligibleCount), getThreadsPerBlock()>>>(getParticles(),
                                                                                           _enhancedMergeData,
                                                                                           getState().grid,
                                                                                           getParameters(),
                                                                                           mergeConfig);
    resolveProposals<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(getParticles(),
                                                                                              _enhancedMergeData);

    refinement::createMergePairs<<<getBlocksPerGridForParticles(eligibleCount), getThreadsPerBlock()>>>(
        _enhancedMergeData);
    cudaDeviceSynchronize();
    const uint32_t pairCount = fromGpu(_enhancedMergeData.pairCount);
    if (pairCount == 0)
    {
        return;
    }
    refinement::executeMerges<<<getBlocksPerGridForParticles(pairCount), getThreadsPerBlock()>>>(getParticles(),
                                                                                                 _enhancedMergeData,
                                                                                                 getParameters());
    refinement::buildCompactionMap<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
        _enhancedMergeData,
        currentCount);
    thrust::exclusive_scan(thrust::device,
                           _enhancedMergeData.compactionMap.data,
                           _enhancedMergeData.compactionMap.data + currentCount,
                           _enhancedMergeData.compactionMap.data);

    refinement::compactParticles<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
        getParticles(),
        _enhancedMergeData,
        currentCount);
    const uint32_t removedCount = fromGpu(&_enhancedMergeData.compactionMap.data[currentCount - 1]);
    const uint32_t newCount = currentCount - removedCount;

    setParticleCount(newCount);
    cudaMemcpy(_refinementData.particlesCount, &newCount, sizeof(uint32_t), cudaMemcpyHostToDevice);
}

refinement::EnhancedMergeData AdaptiveSphSimulation::allocateEnhancedMergeData(uint32_t maxParticleCount)
{
    refinement::EnhancedMergeData data;

    cudaMalloc(&data.criterionValues.data, maxParticleCount * sizeof(float));
    data.criterionValues.size = maxParticleCount;

    cudaMalloc(&data.eligibleParticles.data, maxParticleCount * sizeof(uint32_t));
    data.eligibleParticles.size = maxParticleCount;

    cudaMalloc(&data.eligibleCount, sizeof(uint32_t));

    cudaMalloc(&data.states.data, maxParticleCount * sizeof(refinement::MergeState));
    data.states.size = maxParticleCount;

    const size_t maxPairs = maxParticleCount / 2;
    cudaMalloc(&data.pairs.data, maxPairs * sizeof(refinement::MergePair));
    data.pairs.size = maxPairs;

    cudaMalloc(&data.pairCount, sizeof(uint32_t));

    cudaMalloc(&data.compactionMap.data, maxParticleCount * sizeof(uint32_t));
    data.compactionMap.size = maxParticleCount;

    cudaMalloc(&data.newParticleCount, sizeof(uint32_t));

    return data;
}

void AdaptiveSphSimulation::calculateMergeCriteria(Span<float> criterionValues) const
{
    const float maxMass = _refinementParams.maxMassRatio * getParameters().baseParticleMass;

    if (_refinementParams.criterionType == "interface")
    {
        getCriterionValuesNoGrid<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
            getParticles(),
            criterionValues,
            refinement::interfaceCriterion::MergeCriterionGenerator(maxMass, _refinementParams.interfaceParameters),
            getParameters());
    }
    else if (_refinementParams.criterionType == "vorticity")
    {
        getCriterionValuesWithGrid<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
            getParticles(),
            criterionValues,
            refinement::vorticity::MergeCriterionGenerator(maxMass, _refinementParams.vorticity),
            getState().grid,
            getParameters());
    }
    else if (_refinementParams.criterionType == "curvature")
    {
        getCriterionValuesWithGrid<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
            getParticles(),
            criterionValues,
            refinement::curvature::MergeCriterionGenerator(maxMass, _refinementParams.curvature),
            getState().grid,
            getParameters());
    }
    else
    {
        refinement::getCriterionValues<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
            getParticles(),
            criterionValues,
            refinement::velocity::MergeCriterionGenerator(maxMass, _refinementParams.velocity.merge));
    }
}

void AdaptiveSphSimulation::freeEnhancedMergeData(const refinement::EnhancedMergeData& data)
{
    cudaFree(data.criterionValues.data);
    cudaFree(data.eligibleParticles.data);
    cudaFree(data.eligibleCount);
    cudaFree(data.states.data);
    cudaFree(data.pairs.data);
    cudaFree(data.pairCount);
    cudaFree(data.compactionMap.data);
    cudaFree(data.newParticleCount);
}

void AdaptiveSphSimulation::resetEnhancedMergeData(uint32_t currentParticleCount) const
{
    // Reset counters
    cudaMemset(_enhancedMergeData.eligibleCount, 0, sizeof(uint32_t));
    cudaMemset(_enhancedMergeData.pairCount, 0, sizeof(uint32_t));
    cudaMemset(_enhancedMergeData.newParticleCount, 0, sizeof(uint32_t));
    // Clear state arrays    cudaMemset(_enhancedMergeData.states.data, 0,_enhancedMergeData.states.size * sizeof(refinement::MergeState));
    // Clear criterion values
    cudaMemset(_enhancedMergeData.criterionValues.data, 0, _enhancedMergeData.criterionValues.size * sizeof(float));
    // Clear compaction map up to current particle count
    cudaMemset(_enhancedMergeData.compactionMap.data, 0, currentParticleCount * sizeof(uint32_t));
    // Clear eligible particles array
    cudaMemset(_enhancedMergeData.eligibleParticles.data,
               0,
               _enhancedMergeData.eligibleParticles.size * sizeof(uint32_t));

    // Clear pairs array
    cudaMemset(_enhancedMergeData.pairs.data, 0, _enhancedMergeData.pairs.size * sizeof(refinement::MergePair));
}
}
