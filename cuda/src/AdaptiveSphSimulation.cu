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
      // Initialize CudaMemory objects with RAII
      _criterionValuesSplit(refinementParams.maxParticleCount),
      _particlesIdsToSplit(static_cast<size_t>(refinementParams.maxParticleCount * refinementParams.maxBatchRatio)),
      _particlesSplitCount(1),
      _criterionValuesMerge(refinementParams.maxParticleCount),
      _particlesIdsToMergeFirst(
          static_cast<size_t>(refinementParams.maxParticleCount * refinementParams.maxBatchRatio)),
      _particlesIdsToMergeSecond(
          static_cast<size_t>(refinementParams.maxParticleCount * refinementParams.maxBatchRatio)),
      _removalFlags(refinementParams.maxParticleCount),
      _prefixSums(refinementParams.maxParticleCount),
      _particlesMergeCount(1),
      _particlesIds(refinementParams.maxParticleCount),
      _particlesCount(1),
      // Initialize enhanced merge data
      _enhancedCriterionValues(refinementParams.maxParticleCount),
      _eligibleParticles(refinementParams.maxParticleCount),
      _eligibleCount(1),
      _states(refinementParams.maxParticleCount),
      _pairs(refinementParams.maxParticleCount / 2),
      _pairCount(1),
      _compactionMap(refinementParams.maxParticleCount),
      _newParticleCount(1),
      // Create wrapped data structures
      _refinementData {
          .split {.criterionValues {.data = _criterionValuesSplit.get(), .size = _criterionValuesSplit.size()},
                  .particlesIdsToSplit {.data = _particlesIdsToSplit.get(), .size = _particlesIdsToSplit.size()},
                  .particlesSplitCount {_particlesSplitCount.get()}},
          .merge {.criterionValues {.data = _criterionValuesMerge.get(), .size = _criterionValuesMerge.size()},
                  .particlesIdsToMerge {
                      Span {.data = _particlesIdsToMergeFirst.get(), .size = _particlesIdsToMergeFirst.size()},
                      Span {.data = _particlesIdsToMergeSecond.get(), .size = _particlesIdsToMergeSecond.size()}},
                  .removalFlags {.data = _removalFlags.get(), .size = _removalFlags.size()},
                  .prefixSums {.data = _prefixSums.get(), .size = _prefixSums.size()},
                  .particlesMergeCount {_particlesMergeCount.get()}},
          .particlesIds {.data = _particlesIds.get(), .size = _particlesIds.size()},
          .particlesCount {_particlesCount.get()}
}

      ,
      _enhancedMergeData {
          .criterionValues = {.data = _enhancedCriterionValues.get(), .size = _enhancedCriterionValues.size()},
          .eligibleParticles = {.data = _eligibleParticles.get(), .size = _eligibleParticles.size()},
          .eligibleCount = _eligibleCount.get(),
          .states = {.data = _states.get(), .size = _states.size()},
          .pairs = {.data = _pairs.get(), .size = _pairs.size()},
          .pairCount = _pairCount.get(),
          .compactionMap = {.data = _compactionMap.get(), .size = _compactionMap.size()},
          .newParticleCount = _newParticleCount.get()},
      _targetParticleCount(static_cast<uint32_t>(positions.size()))
{
    const auto initialCount = SphSimulation::getParticlesCount();
    cudaMemcpy(_refinementData.particlesCount, &initialCount, sizeof(uint32_t), cudaMemcpyHostToDevice);
}

void AdaptiveSphSimulation::update(float deltaTime)
{
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
