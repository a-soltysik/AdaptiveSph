#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <vector_types.h>

#include <cstdint>
#include <glm/ext/vector_float4.hpp>
#include <vector>

#include "AdaptiveSphSimulation.cuh"
#include "SphSimulation.cuh"
#include "algorithm/AdaptiveAlgorithm.cuh"
#include "cuda/refinement/RefinementParameters.cuh"
#include "cuda/simulation/Simulation.cuh"
#include "refinement/Common.cuh"
#include "refinement/criteria/InterfaceCriterion.cuh"
#include "refinement/criteria/VelocityCriterion.cuh"
#include "refinement/criteria/VorticityCriterion.cuh"
#include "utils/DeviceView.cuh"

namespace sph::cuda
{

AdaptiveSphSimulation::AdaptiveSphSimulation(const Parameters& initialParameters,
                                             const std::vector<glm::vec4>& positions,
                                             const FluidParticlesDataImportedBuffer& fluidParticleMemory,
                                             const BoundaryParticlesDataImportedBuffer& boundaryParticleMemory,
                                             const physics::StaticBoundaryDomain& boundaryDomain,
                                             const refinement::RefinementParameters& refinementParams,
                                             uint32_t maxBoundaryParticleCapacity)
    : SphSimulation(initialParameters,
                    positions,
                    fluidParticleMemory,
                    boundaryParticleMemory,
                    boundaryDomain,
                    refinementParams.maxParticleCount,
                    maxBoundaryParticleCapacity),
      _refinementParams(refinementParams),
      _refinementData {
          .split = {.criterionValues = {thrust::device_vector<float>(refinementParams.maxParticleCount)},
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
          .particlesCount = DeviceValue<uint32_t>::fromHost(SphSimulation::getFluidParticlesCount())
}
{
}

void AdaptiveSphSimulation::update(float deltaTime)
{
    halfKickVelocities(deltaTime / 2.F);
    updatePositions(deltaTime);

    updateGrid();
    computeDensities();
    computeBoundaryDensityContribution();

    computeExternalAccelerations();
    computePressureAccelerations();
    computeViscosityAccelerations();
    computeSurfaceTensionAccelerations();

    computeBoundaryForces();

    if (_frameCounter == _refinementParams.initialCooldown ||
        (_frameCounter > _refinementParams.initialCooldown && _frameCounter % _refinementParams.cooldown == 0))
    {
        performAdaptiveRefinement();
    }

    halfKickVelocities(deltaTime / 2.F);

    cudaDeviceSynchronize();
    _frameCounter++;
}

void AdaptiveSphSimulation::enableAdaptiveRefinement()
{
    _frameCounter = _refinementParams.initialCooldown;
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
    const auto currentCount = getFluidParticlesCount();
    resetRefinementCounters();
    identifyAndMergeParticles();
    const auto particlesRemovedInLastMerge = currentCount - getFluidParticlesCount();

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
        refinement::getCriterionValues<<<getBlocksPerGridForFluidParticles(), getThreadsPerBlock()>>>(
            getFluidParticles(),
            thrust::raw_pointer_cast(_refinementData.split.criterionValues.data()),
            refinement::interfaceCriterion::SplitCriterionGenerator(minMass,
                                                                    _refinementParams.interfaceParameters,
                                                                    _refinementParams.splitting),
            getGrid().toDevice(),
            getParameters());
    }
    else if (_refinementParams.criterionType == refinement::RefinementParameters::Criterion::Vorticity)
    {
        refinement::getCriterionValues<<<getBlocksPerGridForFluidParticles(), getThreadsPerBlock()>>>(
            getFluidParticles(),
            thrust::raw_pointer_cast(_refinementData.split.criterionValues.data()),
            refinement::vorticity::SplitCriterionGenerator(minMass,
                                                           _refinementParams.vorticity,
                                                           _refinementParams.splitting),
            getGrid().toDevice(),
            getParameters());
    }
    else
    {
        refinement::getCriterionValues<<<getBlocksPerGridForFluidParticles(), getThreadsPerBlock()>>>(
            getFluidParticles(),
            thrust::raw_pointer_cast(_refinementData.split.criterionValues.data()),
            refinement::velocity::SplitCriterionGenerator(minMass,
                                                          _refinementParams.velocity.split,
                                                          _refinementParams.splitting),
            getGrid().toDevice(),
            getParameters());
    }

    refinement::findTopParticlesToSplit(getFluidParticles(),
                                        _refinementData,
                                        _refinementParams,
                                        thrust::greater<float> {});

    const auto particlesToSplitCount =
        std::min(_refinementData.split.particlesSplitCount.toHost(), removedParticles / 12);

    if (particlesToSplitCount == 0)
    {
        return;
    }

    refinement::splitParticles<<<getBlocksPerGridForParticles(particlesToSplitCount), getThreadsPerBlock()>>>(
        getFluidParticles(),
        refinement::RefinementDataView {_refinementData},
        _refinementParams.splitting,
        _refinementParams.maxParticleCount);

    setParticleCount(_refinementData.particlesCount.toHost());
}

void AdaptiveSphSimulation::identifyAndMergeParticles()
{
    if (getFluidParticlesCount() <= 1)
    {
        return;
    }

    const auto currentCount = getFluidParticlesCount();

    calculateMergeCriteria(thrust::raw_pointer_cast(_refinementData.merge.criterionValues.data()));

    refinement::findTopParticlesToMerge(getFluidParticles(),
                                        _refinementData,
                                        _refinementParams,
                                        thrust::less<float> {});

    const auto particlesToMergeCount = _refinementData.merge.eligibleCount.toHost();

    if (particlesToMergeCount == 0)
    {
        return;
    }

    refinement::identifyMergeCandidates<<<getBlocksPerGridForParticles(particlesToMergeCount), getThreadsPerBlock()>>>(
        getFluidParticles(),
        refinement::RefinementDataView {_refinementData},
        getGrid().toDevice(),
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
        getFluidParticles(),
        refinement::RefinementDataView {_refinementData},
        getParameters());

    thrust::exclusive_scan(
        thrust::device,
        reinterpret_cast<uint32_t*>(thrust::raw_pointer_cast(_refinementData.merge.removalFlags.data())),
        reinterpret_cast<uint32_t*>(thrust::raw_pointer_cast(_refinementData.merge.removalFlags.data())) + currentCount,
        _refinementData.merge.prefixSums.data());

    refinement::compactParticles<<<getBlocksPerGridForParticles(currentCount), getThreadsPerBlock()>>>(
        getFluidParticles(),
        refinement::RefinementDataView {_refinementData},
        currentCount);

    setParticleCount(_refinementData.particlesCount.toHost());
}

void AdaptiveSphSimulation::calculateMergeCriteria(float* criterionValues)
{
    const float maxMass = _refinementParams.maxMassRatio * getParameters().baseParticleMass;

    if (_refinementParams.criterionType == refinement::RefinementParameters::Criterion::Interface)
    {
        refinement::getCriterionValues<<<getBlocksPerGridForFluidParticles(), getThreadsPerBlock()>>>(
            getFluidParticles(),
            criterionValues,
            refinement::interfaceCriterion::MergeCriterionGenerator(maxMass, _refinementParams.interfaceParameters),
            getGrid().toDevice(),
            getParameters());
    }
    else if (_refinementParams.criterionType == refinement::RefinementParameters::Criterion::Vorticity)
    {
        refinement::getCriterionValues<<<getBlocksPerGridForFluidParticles(), getThreadsPerBlock()>>>(
            getFluidParticles(),
            criterionValues,
            refinement::vorticity::MergeCriterionGenerator(maxMass, _refinementParams.vorticity),
            getGrid().toDevice(),
            getParameters());
    }
    else
    {
        refinement::getCriterionValues<<<getBlocksPerGridForFluidParticles(), getThreadsPerBlock()>>>(
            getFluidParticles(),
            criterionValues,
            refinement::velocity::MergeCriterionGenerator(maxMass, _refinementParams.velocity.merge),
            getGrid().toDevice(),
            getParameters());
    }
}
}
