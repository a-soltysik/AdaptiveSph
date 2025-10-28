#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <vector_types.h>

#include <cstdint>
#include <glm/ext/vector_float4.hpp>
#include <vector>

#include "AdaptiveSphSimulation.cuh"
#include "algorithm/adaptive/AdaptiveAlgorithm.cuh"
#include "cuda/Simulation.cuh"
#include "cuda/refinement/RefinementParameters.cuh"
#include "refinement/criteria/CurvatureCriterion.cuh"
#include "refinement/criteria/InterfaceCriterion.cuh"
#include "refinement/criteria/VelocityCriterion.cuh"
#include "refinement/criteria/VorticityCriterion.cuh"
#include "simulation/adaptive/SphSimulation.cuh"
#include "utils/Utils.cuh"

namespace sph::cuda
{

AdaptiveSphSimulation::AdaptiveSphSimulation(const Parameters& initialParameters,
                                             const std::vector<glm::vec4>& positions,
                                             const ParticlesDataBuffer& memory,
                                             const refinement::RefinementParameters& refinementParams)
    : SphSimulation(initialParameters, positions, memory, refinementParams.maxParticleCount),
      _criterionValuesSplit(refinementParams.maxParticleCount),
      _particlesIdsToSplit(
          static_cast<size_t>(static_cast<float>(refinementParams.maxParticleCount) * refinementParams.maxBatchRatio)),
      _particlesSplitCount(1),
      _particlesIds(refinementParams.maxParticleCount),
      _particlesCount(1),
      _mergeCriterionValues(refinementParams.maxParticleCount),
      _mergeEligibleParticles(refinementParams.maxParticleCount),
      _mergeEligibleCount(1),
      _mergeCandidates(refinementParams.maxParticleCount),
      _mergePairs(refinementParams.maxParticleCount * 2),
      _mergeCount(1),
      _mergeRemovalFlags(refinementParams.maxParticleCount),
      _mergePrefixSums(refinementParams.maxParticleCount),
      _refinementParams(refinementParams),
      _refinementData {
          .split = {.criterionValues = {_criterionValuesSplit.get(), _criterionValuesSplit.size()},
                    .particlesIdsToSplit = {_particlesIdsToSplit.get(), _particlesIdsToSplit.size()},
                    .particlesSplitCount = {_particlesSplitCount.get()}},
          .merge = {.criterionValues = {_mergeCriterionValues.get(), _mergeCriterionValues.size()},
                    .eligibleParticles = {_mergeEligibleParticles.get(), _mergeEligibleParticles.size()},
                    .eligibleCount = _mergeEligibleCount.get(),
                    .mergeCandidates = {_mergeCandidates.get(), _mergeCandidates.size()},
                    .mergePairs = {_mergePairs.get(), _mergePairs.size()},
                    .removalFlags = {_mergeRemovalFlags.get(), _mergeRemovalFlags.size()},
                    .prefixSums = {_mergePrefixSums.get(), _mergePrefixSums.size()},
                    .mergeCount = _mergeCount.get()},
          .particlesIds = {_particlesIds.get(), _particlesIds.size()},
          .particlesCount = {_particlesCount.get()}
},
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
    const auto currentCount = getParticlesCount();
    resetRefinementCounters();
    identifyAndMergeParticles();
    const auto particlesRemovedInLastMerge = currentCount - getParticlesCount();

    if (particlesRemovedInLastMerge > 0)
    {
        identifyAndSplitParticles(particlesRemovedInLastMerge);
    }
}

void AdaptiveSphSimulation::resetRefinementCounters() const
{
    static constexpr uint32_t zero = 0;

    cudaMemcpy(_refinementData.split.particlesSplitCount, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(_refinementData.merge.eligibleCount, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(_refinementData.merge.mergeCount, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(_refinementData.split.criterionValues.data(),
               0,
               _refinementData.split.criterionValues.size() * sizeof(float));
    cudaMemset(_refinementData.merge.criterionValues.data(),
               0,
               _refinementData.merge.criterionValues.size() * sizeof(float));
    cudaMemset(_refinementData.particlesIds.data(), 0, _refinementData.particlesIds.size() * sizeof(uint32_t));

    cudaMemset(_refinementData.merge.removalFlags.data(),
               0,
               _refinementData.merge.removalFlags.size() * sizeof(uint32_t));
    cudaMemset(_refinementData.merge.prefixSums.data(), 0, _refinementData.merge.prefixSums.size() * sizeof(uint32_t));

    cudaMemset(_refinementData.split.particlesIdsToSplit.data(),
               0,
               _refinementData.split.particlesIdsToSplit.size() * sizeof(uint32_t));
    cudaMemset(_refinementData.merge.eligibleParticles.data(),
               0,
               _refinementData.merge.eligibleParticles.size() * sizeof(uint32_t));
    cudaMemset(_refinementData.merge.mergeCandidates.data(),
               0,
               _refinementData.merge.mergeCandidates.size() * sizeof(uint32_t));
    cudaMemset(_refinementData.merge.mergePairs.data(), 0, _refinementData.merge.mergePairs.size() * sizeof(uint32_t));

    cudaMemset(_refinementData.merge.removalFlags.data(),
               0,
               _refinementData.merge.removalFlags.size() * sizeof(uint32_t));
    cudaMemset(_refinementData.merge.mergeCount, 0, sizeof(uint32_t));
    cudaMemset(_refinementData.merge.eligibleCount, 0, sizeof(uint32_t));
}

void AdaptiveSphSimulation::identifyAndSplitParticles(uint32_t removedParticles)
{
    const float minMass = _refinementParams.minMassRatio * getParameters().baseParticleMass;

    if (_refinementParams.criterionType == refinement::RefinementParameters::Criterion::Interface)
    {
        refinement::getCriterionValues<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
            getParticles(),
            _refinementData.split.criterionValues,
            refinement::interfaceCriterion::SplitCriterionGenerator(minMass, _refinementParams.interfaceParameters),
            getState().grid,
            getParameters());
    }
    else if (_refinementParams.criterionType == refinement::RefinementParameters::Criterion::Vorticity)
    {
        refinement::getCriterionValues<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
            getParticles(),
            _refinementData.split.criterionValues,
            refinement::vorticity::SplitCriterionGenerator(minMass, _refinementParams.vorticity),
            getState().grid,
            getParameters());
    }
    else if (_refinementParams.criterionType == refinement::RefinementParameters::Criterion::Curvature)
    {
        refinement::getCriterionValues<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
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
            refinement::velocity::SplitCriterionGenerator(minMass, _refinementParams.velocity.split),
            getState().grid,
            getParameters());
    }

    refinement::findTopParticlesToSplit(getParticles(), _refinementData, _refinementParams, thrust::greater<float> {});

    const auto particlesToSplitCount =
        std::min(fromGpu(_refinementData.split.particlesSplitCount), removedParticles / 12);

    if (particlesToSplitCount == 0)
    {
        return;
    }

    refinement::splitParticles<<<getBlocksPerGridForParticles(particlesToSplitCount), getThreadsPerBlock()>>>(
        getParticles(),
        _refinementData,
        _refinementParams.splitting,
        _refinementParams.maxParticleCount);

    setParticleCount(fromGpu(_refinementData.particlesCount));
}

void AdaptiveSphSimulation::identifyAndMergeParticles()
{
    if (getParticlesCount() <= 1)
    {
        return;
    }

    const auto currentCount = getParticlesCount();

    calculateMergeCriteria(_refinementData.merge.criterionValues);

    refinement::findTopParticlesToMerge(getParticles(), _refinementData, _refinementParams, thrust::less<float> {});

    const auto particlesToMergeCount = fromGpu(_refinementData.merge.eligibleCount);

    if (particlesToMergeCount == 0)
    {
        return;
    }

    refinement::identifyMergeCandidates<<<getBlocksPerGridForParticles(particlesToMergeCount), getThreadsPerBlock()>>>(
        getParticles(),
        _refinementData.merge,
        getState().grid,
        getParameters(),
        _refinementParams);

    refinement::resolveMergePairs<<<getBlocksPerGridForParticles(currentCount), getThreadsPerBlock()>>>(
        _refinementData.merge,
        currentCount);

    const auto mergeCount = fromGpu(_refinementData.merge.mergeCount);
    if (mergeCount == 0)
    {
        return;
    }

    refinement::performMerges<<<getBlocksPerGridForParticles(mergeCount), getThreadsPerBlock()>>>(getParticles(),
                                                                                                  _refinementData.merge,
                                                                                                  getParameters());
    thrust::exclusive_scan(thrust::device,
                           reinterpret_cast<uint32_t*>(_refinementData.merge.removalFlags.data()),
                           reinterpret_cast<uint32_t*>(_refinementData.merge.removalFlags.data()) + currentCount,
                           _refinementData.merge.prefixSums.data());

    refinement::compactParticles<<<getBlocksPerGridForParticles(currentCount), getThreadsPerBlock()>>>(
        getParticles(),
        _refinementData.merge,
        currentCount);

    refinement::updateParticleCount<<<1, 1>>>(_refinementData, currentCount);

    setParticleCount(fromGpu(_refinementData.particlesCount));
}

void AdaptiveSphSimulation::calculateMergeCriteria(std::span<float> criterionValues) const
{
    const float maxMass = _refinementParams.maxMassRatio * getParameters().baseParticleMass;

    if (_refinementParams.criterionType == refinement::RefinementParameters::Criterion::Interface)
    {
        refinement::getCriterionValues<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
            getParticles(),
            criterionValues,
            refinement::interfaceCriterion::MergeCriterionGenerator(maxMass, _refinementParams.interfaceParameters),
            getState().grid,
            getParameters());
    }
    else if (_refinementParams.criterionType == refinement::RefinementParameters::Criterion::Vorticity)
    {
        refinement::getCriterionValues<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
            getParticles(),
            criterionValues,
            refinement::vorticity::MergeCriterionGenerator(maxMass, _refinementParams.vorticity),
            getState().grid,
            getParameters());
    }
    else if (_refinementParams.criterionType == refinement::RefinementParameters::Criterion::Curvature)
    {
        refinement::getCriterionValues<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
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
            refinement::velocity::MergeCriterionGenerator(maxMass, _refinementParams.velocity.merge),
            getState().grid,
            getParameters());
    }
}

}
