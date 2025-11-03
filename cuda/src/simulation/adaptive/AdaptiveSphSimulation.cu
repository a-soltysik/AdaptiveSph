#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <vector_types.h>

#include <cstdint>
#include <glm/ext/vector_float4.hpp>
#include <vector>

#include "../SphSimulation.cuh"
#include "AdaptiveSphSimulation.cuh"
#include "algorithm/adaptive/AdaptiveAlgorithm.cuh"
#include "cuda/Simulation.cuh"
#include "cuda/refinement/RefinementParameters.cuh"
#include "refinement/criteria/CurvatureCriterion.cuh"
#include "refinement/criteria/InterfaceCriterion.cuh"
#include "refinement/criteria/VelocityCriterion.cuh"
#include "refinement/criteria/VorticityCriterion.cuh"
#include "utils/Utils.cuh"

namespace sph::cuda
{

AdaptiveSphSimulation::AdaptiveSphSimulation(const Parameters& initialParameters,
                                             const std::vector<glm::vec4>& positions,
                                             const ParticlesDataBuffer& memory,
                                             const refinement::RefinementParameters& refinementParams)
    : SphSimulation(initialParameters, positions, memory, refinementParams.maxParticleCount),
      _refinementParams(refinementParams),
      _refinementData {
          .split = {.criterionValues = {thrust::device_vector<float>(static_cast<size_t>(
                        static_cast<float>(refinementParams.maxParticleCount) * refinementParams.maxBatchRatio))},
                    .particlesIdsToSplit = {thrust::device_vector<uint32_t>(refinementParams.maxParticleCount)},
                    .particlesSplitCount = DeviceValue<uint32_t>::fromHost(0)},
          .merge = {.criterionValues = {thrust::device_vector<float>(refinementParams.maxParticleCount)},
                    .eligibleParticles = {thrust::device_vector<uint32_t>(refinementParams.maxParticleCount)},
                    .eligibleCount = DeviceValue<uint32_t>::fromHost(0),
                    .mergeCandidates = {thrust::device_vector<uint32_t>(refinementParams.maxParticleCount)},
                    .mergePairs = {thrust::device_vector<uint32_t>(refinementParams.maxParticleCount * 2)},
                    .removalFlags = {thrust::device_vector<refinement::RefinementData::RemovalState>(
                        refinementParams.maxParticleCount)},
                    .prefixSums = {thrust::device_vector<uint32_t>(refinementParams.maxParticleCount)},
                    .mergeCount = DeviceValue<uint32_t>::fromHost(0)},
          .particlesIds = {thrust::device_vector<uint32_t>(refinementParams.maxParticleCount)},
          .particlesCount = DeviceValue<uint32_t>::fromHost(SphSimulation::getParticlesCount())
},
      _targetParticleCount(static_cast<uint32_t>(positions.size()))
{
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
    setParticleCount(_refinementData.particlesCount.toHost());
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

void AdaptiveSphSimulation::resetRefinementCounters()
{
    thrust::fill(_refinementData.split.criterionValues.begin(), _refinementData.split.criterionValues.end(), 0);
    thrust::fill(_refinementData.split.particlesIdsToSplit.begin(), _refinementData.split.particlesIdsToSplit.end(), 0);
    _refinementData.split.particlesSplitCount = 0;

    thrust::fill(_refinementData.merge.criterionValues.begin(), _refinementData.merge.criterionValues.end(), 0);
    thrust::fill(_refinementData.merge.removalFlags.begin(),
                 _refinementData.merge.removalFlags.end(),
                 refinement::RefinementData::RemovalState::Keep);
    thrust::fill(_refinementData.merge.prefixSums.begin(), _refinementData.merge.prefixSums.end(), 0);
    thrust::fill(_refinementData.merge.eligibleParticles.begin(), _refinementData.merge.eligibleParticles.end(), 0);
    thrust::fill(_refinementData.merge.mergeCandidates.begin(), _refinementData.merge.mergeCandidates.end(), 0);
    thrust::fill(_refinementData.merge.mergePairs.begin(), _refinementData.merge.mergePairs.end(), 0);
    _refinementData.merge.eligibleCount = 0;
    _refinementData.merge.mergeCount = 0;

    thrust::fill(_refinementData.particlesIds.begin(), _refinementData.particlesIds.end(), 0);
}

void AdaptiveSphSimulation::identifyAndSplitParticles(uint32_t removedParticles)
{
    const float minMass = _refinementParams.minMassRatio * getParameters().baseParticleMass;

    if (_refinementParams.criterionType == refinement::RefinementParameters::Criterion::Interface)
    {
        refinement::getCriterionValues<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
            getParticles(),
            thrust::raw_pointer_cast(_refinementData.split.criterionValues.data()),
            refinement::interfaceCriterion::SplitCriterionGenerator(minMass, _refinementParams.interfaceParameters),
            getGrid(),
            getParameters());
    }
    else if (_refinementParams.criterionType == refinement::RefinementParameters::Criterion::Vorticity)
    {
        refinement::getCriterionValues<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
            getParticles(),
            thrust::raw_pointer_cast(_refinementData.split.criterionValues.data()),
            refinement::vorticity::SplitCriterionGenerator(minMass, _refinementParams.vorticity),
            getGrid(),
            getParameters());
    }
    else if (_refinementParams.criterionType == refinement::RefinementParameters::Criterion::Curvature)
    {
        refinement::getCriterionValues<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
            getParticles(),
            thrust::raw_pointer_cast(_refinementData.split.criterionValues.data()),
            refinement::curvature::SplitCriterionGenerator(minMass, _refinementParams.curvature),
            getGrid(),
            getParameters());
    }
    else
    {
        refinement::getCriterionValues<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
            getParticles(),
            thrust::raw_pointer_cast(_refinementData.split.criterionValues.data()),
            refinement::velocity::SplitCriterionGenerator(minMass, _refinementParams.velocity.split),
            getGrid(),
            getParameters());
    }

    refinement::findTopParticlesToSplit(getParticles(), _refinementData, _refinementParams, thrust::greater<float> {});

    const auto particlesToSplitCount =
        std::min(_refinementData.split.particlesSplitCount.toHost(), removedParticles / 12);

    if (particlesToSplitCount == 0)
    {
        return;
    }

    refinement::splitParticles<<<getBlocksPerGridForParticles(particlesToSplitCount), getThreadsPerBlock()>>>(
        getParticles(),
        refinement::RefinementDataView {_refinementData},
        _refinementParams.splitting,
        _refinementParams.maxParticleCount);

    setParticleCount(_refinementData.particlesCount.toHost());
}

void AdaptiveSphSimulation::identifyAndMergeParticles()
{
    if (getParticlesCount() <= 1)
    {
        return;
    }

    const auto currentCount = getParticlesCount();

    calculateMergeCriteria(thrust::raw_pointer_cast(_refinementData.merge.criterionValues.data()));

    refinement::findTopParticlesToMerge(getParticles(), _refinementData, _refinementParams, thrust::less<float> {});

    const auto particlesToMergeCount = _refinementData.merge.eligibleCount.toHost();

    if (particlesToMergeCount == 0)
    {
        return;
    }

    refinement::identifyMergeCandidates<<<getBlocksPerGridForParticles(particlesToMergeCount), getThreadsPerBlock()>>>(
        getParticles(),
        refinement::RefinementDataView {_refinementData},
        getGrid(),
        getParameters(),
        _refinementParams);

    refinement::resolveMergePairs<<<getBlocksPerGridForParticles(currentCount), getThreadsPerBlock()>>>(
        refinement::RefinementDataView {_refinementData},
        currentCount);

    const auto mergeCount = _refinementData.merge.mergeCount.toHost();
    if (mergeCount == 0)
    {
        return;
    }

    refinement::performMerges<<<getBlocksPerGridForParticles(mergeCount), getThreadsPerBlock()>>>(
        getParticles(),
        refinement::RefinementDataView {_refinementData},
        getParameters());
    thrust::exclusive_scan(
        thrust::device,
        reinterpret_cast<uint32_t*>(thrust::raw_pointer_cast(_refinementData.merge.removalFlags.data())),
        reinterpret_cast<uint32_t*>(thrust::raw_pointer_cast(_refinementData.merge.removalFlags.data())) + currentCount,
        _refinementData.merge.prefixSums.data());

    refinement::compactParticles<<<getBlocksPerGridForParticles(currentCount), getThreadsPerBlock()>>>(
        getParticles(),
        refinement::RefinementDataView {_refinementData},
        currentCount);

    setParticleCount(_refinementData.particlesCount.toHost());
}

void AdaptiveSphSimulation::calculateMergeCriteria(float* criterionValues) const
{
    const float maxMass = _refinementParams.maxMassRatio * getParameters().baseParticleMass;

    if (_refinementParams.criterionType == refinement::RefinementParameters::Criterion::Interface)
    {
        refinement::getCriterionValues<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
            getParticles(),
            criterionValues,
            refinement::interfaceCriterion::MergeCriterionGenerator(maxMass, _refinementParams.interfaceParameters),
            getGrid(),
            getParameters());
    }
    else if (_refinementParams.criterionType == refinement::RefinementParameters::Criterion::Vorticity)
    {
        refinement::getCriterionValues<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
            getParticles(),
            criterionValues,
            refinement::vorticity::MergeCriterionGenerator(maxMass, _refinementParams.vorticity),
            getGrid(),
            getParameters());
    }
    else if (_refinementParams.criterionType == refinement::RefinementParameters::Criterion::Curvature)
    {
        refinement::getCriterionValues<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
            getParticles(),
            criterionValues,
            refinement::curvature::MergeCriterionGenerator(maxMass, _refinementParams.curvature),
            getGrid(),
            getParameters());
    }
    else
    {
        refinement::getCriterionValues<<<SphSimulation::getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(
            getParticles(),
            criterionValues,
            refinement::velocity::MergeCriterionGenerator(maxMass, _refinementParams.velocity.merge),
            getGrid(),
            getParameters());
    }
}
}
