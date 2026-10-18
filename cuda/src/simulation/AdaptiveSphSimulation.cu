#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <thrust/execution_policy.h>
#include <vector_types.h>

#include <cstdint>
#include <cuda/std/functional>
#include <glm/ext/vector_float4.hpp>
#include <utility>
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
#include "utils/CudaError.cuh"

namespace sph::cuda
{

AdaptiveSphSimulation::AdaptiveSphSimulation(const Parameters& initialParameters,
                                             const std::vector<glm::vec4>& positions,
                                             const FluidParticlesDataImportedBuffer& fluidParticleMemory,
                                             const BoundaryParticlesDataImportedBuffer& boundaryParticleMemory,
                                             const physics::StaticBoundaryDomain& boundaryDomain,
                                             const refinement::RefinementParameters& refinementParams,
                                             uint32_t maxBoundaryParticleCapacity,
                                             const ScenarioParameters& scenarioParams)
    : SphSimulation(initialParameters,
                    positions,
                    fluidParticleMemory,
                    boundaryParticleMemory,
                    boundaryDomain,
                    refinementParams.maxParticleCount,
                    maxBoundaryParticleCapacity,
                    scenarioParams,
                    1.4F),
      _refinementParams(refinementParams),
      _refinementData {
          .split = {.criterionValues = {thrust::device_vector<float>(refinementParams.maxParticleCount)},
                    .particlesIdsToSplit = {thrust::device_vector<uint32_t>(refinementParams.maxParticleCount)},
                    .particlesSplitCount =
                        DeviceBuffer<uint32_t>::fromValue(0, "AdaptiveSphSimulation: initialize split-particle count")},
          .merge = {.criterionValues = {thrust::device_vector<float>(refinementParams.maxParticleCount)},
                    .eligibleParticles = {thrust::device_vector<uint32_t>(refinementParams.maxParticleCount)},
                    .eligibleCount = DeviceBuffer<uint32_t>::fromValue(
                        0, "AdaptiveSphSimulation: initialize eligible-particle count"),
                    .mergeCandidates = {thrust::device_vector<uint32_t>(refinementParams.maxParticleCount)},
                    .mergePairs = {thrust::device_vector<uint32_t>(refinementParams.maxParticleCount * 2)},
                    .removalFlags = {thrust::device_vector<refinement::RefinementData::RemovalState>(
                        refinementParams.maxParticleCount)},
                    .mergeCount =
                        DeviceBuffer<uint32_t>::fromValue(0, "AdaptiveSphSimulation: initialize merge count")},
          .particlesIds = {thrust::device_vector<uint32_t>(refinementParams.maxParticleCount)},
          .particlesCount = DeviceBuffer<uint32_t>::fromValue(SphSimulation::getFluidParticlesCount(),
                                                              "AdaptiveSphSimulation: initialize particle count")
}
{
}

void AdaptiveSphSimulation::update(float deltaTime)
{
    _deltaTime = deltaTime;

    halfKickVelocities(deltaTime / 2.F);
    updatePositions(deltaTime);

    auto gridNeedsRefresh = true;
    const auto scheduledRefinement =
        _frameCounter == _refinementParams.initialCooldown ||
        (_frameCounter > _refinementParams.initialCooldown && _frameCounter % _refinementParams.cooldown == 0);
    if (std::exchange(_refinementRequested, false) || scheduledRefinement)
    {
        gridNeedsRefresh = performAdaptiveRefinement();
    }

    if (gridNeedsRefresh)
    {
        updateGrid();
        computeDensities();
        computeBoundaryDensityContribution();
    }

    computeExternalAccelerations();
    computePressureAccelerations();
    computeViscosityAccelerations();
    computeSurfaceTensionAccelerations();

    computeBoundaryForces();

    halfKickVelocities(deltaTime / 2.F);

    detail::checkCuda(cudaDeviceSynchronize(), "AdaptiveSphSimulation::update: synchronize completed timestep");
    _frameCounter++;
    _accumulatedSimulationTime += deltaTime;
}

void AdaptiveSphSimulation::requestAdaptiveRefinement()
{
    _refinementRequested = true;
}

auto AdaptiveSphSimulation::takeAdaptiveRefinementPass() -> std::optional<AdaptiveRefinementPass>
{
    const auto refinementPass = _lastRefinementPass;
    _lastRefinementPass.reset();
    return refinementPass;
}

auto AdaptiveSphSimulation::getBlocksPerGridForParticles(uint32_t count) const -> dim3
{
    return {(count + getThreadsPerBlock() - 1) / getThreadsPerBlock()};
}

auto AdaptiveSphSimulation::performAdaptiveRefinement() -> bool
{
    const auto currentCount = getFluidParticlesCount();

    updateGrid();
    computeDensities();
    computeBoundaryDensityContribution();

    resetRefinementCounters();
    identifyAndMergeParticles();

    const auto particlesRemovedInLastMerge = currentCount - getFluidParticlesCount();
    const auto countAfterMerge = getFluidParticlesCount();
    auto gridNeedsRefresh = particlesRemovedInLastMerge > 0;

    const auto availableParticleCapacity = _refinementParams.maxParticleCount - countAfterMerge;
    if (particlesRemovedInLastMerge == 0 || availableParticleCapacity >= refinement::constants::splitChildCount)
    {
        if (particlesRemovedInLastMerge > 0)
        {
            updateGrid();
            computeDensities();
            computeBoundaryDensityContribution();
            gridNeedsRefresh = false;
        }
        gridNeedsRefresh = identifyAndSplitParticles();
    }

    const auto finalCount = getFluidParticlesCount();
    _lastRefinementPass = AdaptiveRefinementPass {
        .particlesBefore = currentCount,
        .particlesAfter = finalCount,
        .mergedPairs = particlesRemovedInLastMerge,
        .splitParents = (finalCount - countAfterMerge) / refinement::constants::splitChildCount};

    return gridNeedsRefresh;
}

void AdaptiveSphSimulation::resetRefinementCounters()
{
    thrust::fill(_refinementData.split.criterionValues.begin(), _refinementData.split.criterionValues.end(), 0);
    thrust::fill(_refinementData.split.particlesIdsToSplit.begin(), _refinementData.split.particlesIdsToSplit.end(), 0);
    _refinementData.split.particlesSplitCount.copyValueFromHost(
        0,
        "AdaptiveSphSimulation::resetRefinementCounters: reset split count");

    thrust::fill(_refinementData.merge.criterionValues.begin(), _refinementData.merge.criterionValues.end(), 0);
    thrust::fill(_refinementData.merge.removalFlags.begin(),
                 _refinementData.merge.removalFlags.end(),
                 refinement::RefinementData::RemovalState::Keep);
    thrust::fill(_refinementData.merge.eligibleParticles.begin(), _refinementData.merge.eligibleParticles.end(), 0);
    thrust::fill(_refinementData.merge.mergeCandidates.begin(),
                 _refinementData.merge.mergeCandidates.end(),
                 refinement::constants::noCandidate);
    thrust::fill(_refinementData.merge.mergePairs.begin(), _refinementData.merge.mergePairs.end(), 0);
    _refinementData.merge.eligibleCount.copyValueFromHost(
        0,
        "AdaptiveSphSimulation::resetRefinementCounters: reset eligible count");
    _refinementData.merge.mergeCount.copyValueFromHost(
        0,
        "AdaptiveSphSimulation::resetRefinementCounters: reset merge count");

    thrust::fill(_refinementData.particlesIds.begin(), _refinementData.particlesIds.end(), 0);
}

auto AdaptiveSphSimulation::identifyAndSplitParticles() -> bool
{
    const auto currentCount = getFluidParticlesCount();
    if (currentCount > _refinementParams.maxParticleCount ||
        _refinementParams.maxParticleCount - currentCount < refinement::constants::splitChildCount)
    {
        return false;
    }

    const float minMass = _refinementParams.minMassRatio * getParameters().baseParticleMass;

    if (_refinementParams.criterionType == refinement::RefinementParameters::Criterion::Interface)
    {
        refinement::getCriterionValues<<<getBlocksPerGridForFluidParticles(), getThreadsPerBlock()>>>(
            getFluidParticles(),
            thrust::raw_pointer_cast(_refinementData.split.criterionValues.data()),
            refinement::interfaceCriterion::SplitCriterionGenerator(minMass, _refinementParams.interfaceParameters),
            getGrid().toDevice(),
            getParameters());
        detail::checkCudaLastError("AdaptiveSphSimulation::identifyAndSplitParticles: launch interface criterion");
    }
    else if (_refinementParams.criterionType == refinement::RefinementParameters::Criterion::Vorticity)
    {
        refinement::getCriterionValues<<<getBlocksPerGridForFluidParticles(), getThreadsPerBlock()>>>(
            getFluidParticles(),
            thrust::raw_pointer_cast(_refinementData.split.criterionValues.data()),
            refinement::vorticity::SplitCriterionGenerator(minMass, _refinementParams.vorticity),
            getGrid().toDevice(),
            getParameters());
        detail::checkCudaLastError("AdaptiveSphSimulation::identifyAndSplitParticles: launch vorticity criterion");
    }
    else
    {
        refinement::getCriterionValues<<<getBlocksPerGridForFluidParticles(), getThreadsPerBlock()>>>(
            getFluidParticles(),
            thrust::raw_pointer_cast(_refinementData.split.criterionValues.data()),
            refinement::velocity::SplitCriterionGenerator(minMass, _refinementParams.velocity.split),
            getGrid().toDevice(),
            getParameters());
        detail::checkCudaLastError("AdaptiveSphSimulation::identifyAndSplitParticles: launch velocity criterion");
    }

    refinement::findTopParticlesToSplit(getFluidParticles(),
                                        _refinementData,
                                        _refinementParams,
                                        ::cuda::std::greater<float> {});

    const auto candidatesCount = _refinementData.split.particlesSplitCount.copyValueToHost(
        "AdaptiveSphSimulation::identifyAndSplitParticles: read split count");
    const auto availableParticleCapacity = _refinementParams.maxParticleCount - currentCount;
    const auto maxSplitCount = availableParticleCapacity / refinement::constants::splitChildCount;
    const auto particlesToSplitCount = std::min(candidatesCount, maxSplitCount);

    if (particlesToSplitCount == 0)
    {
        return false;
    }

    _refinementData.split.particlesSplitCount.copyValueFromHost(
        particlesToSplitCount,
        "AdaptiveSphSimulation::identifyAndSplitParticles: copy split count to device");
    refinement::splitParticles<<<getBlocksPerGridForParticles(particlesToSplitCount), getThreadsPerBlock()>>>(
        getFluidParticles(),
        refinement::RefinementDataView {_refinementData},
        _refinementParams.splitting,
        _refinementParams.maxParticleCount);
    detail::checkCudaLastError("AdaptiveSphSimulation::identifyAndSplitParticles: launch split kernel");

    detail::checkCuda(cudaDeviceSynchronize(), "AdaptiveSphSimulation::identifyAndSplitParticles: synchronize split");
    setParticleCount(_refinementData.particlesCount.copyValueToHost(
        "AdaptiveSphSimulation::identifyAndSplitParticles: read particle count"));
    return getFluidParticlesCount() != currentCount;
}

void AdaptiveSphSimulation::identifyAndMergeParticles()
{
    if (getFluidParticlesCount() <= 1)
    {
        return;
    }

    const auto currentCount = getFluidParticlesCount();

    calculateMergeCriteria(thrust::raw_pointer_cast(_refinementData.merge.criterionValues.data()));

    refinement::findTopParticlesToMerge(getFluidParticles(), _refinementData, _refinementParams);

    const auto particlesToMergeCount = _refinementData.merge.eligibleCount.copyValueToHost(
        "AdaptiveSphSimulation::identifyAndMergeParticles: read eligible count");

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
    detail::checkCudaLastError("AdaptiveSphSimulation::identifyAndMergeParticles: launch candidate kernel");

    refinement::resolveMergePairs<<<getBlocksPerGridForParticles(currentCount), getThreadsPerBlock()>>>(
        refinement::RefinementDataView {_refinementData},
        currentCount);
    detail::checkCudaLastError("AdaptiveSphSimulation::identifyAndMergeParticles: launch pair-resolution kernel");

    const auto mergeCount = _refinementData.merge.mergeCount.copyValueToHost(
        "AdaptiveSphSimulation::identifyAndMergeParticles: read merge count");

    if (mergeCount == 0)
    {
        return;
    }

    refinement::performMerges<<<getBlocksPerGridForParticles(mergeCount), getThreadsPerBlock()>>>(
        getFluidParticles(),
        refinement::RefinementDataView {_refinementData},
        getParameters());
    detail::checkCudaLastError("AdaptiveSphSimulation::identifyAndMergeParticles: launch merge kernel");

    const auto compactedCount = refinement::compactParticles(getFluidParticles(), _refinementData);
    setParticleCount(compactedCount);
    _refinementData.particlesCount.copyValueFromHost(
        compactedCount,
        "AdaptiveSphSimulation::identifyAndMergeParticles: copy compacted count");
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
        detail::checkCudaLastError("AdaptiveSphSimulation::calculateMergeCriteria: launch interface criterion");
    }
    else if (_refinementParams.criterionType == refinement::RefinementParameters::Criterion::Vorticity)
    {
        refinement::getCriterionValues<<<getBlocksPerGridForFluidParticles(), getThreadsPerBlock()>>>(
            getFluidParticles(),
            criterionValues,
            refinement::vorticity::MergeCriterionGenerator(maxMass, _refinementParams.vorticity),
            getGrid().toDevice(),
            getParameters());
        detail::checkCudaLastError("AdaptiveSphSimulation::calculateMergeCriteria: launch vorticity criterion");
    }
    else
    {
        refinement::getCriterionValues<<<getBlocksPerGridForFluidParticles(), getThreadsPerBlock()>>>(
            getFluidParticles(),
            criterionValues,
            refinement::velocity::MergeCriterionGenerator(maxMass, _refinementParams.velocity.merge),
            getGrid().toDevice(),
            getParameters());
        detail::checkCudaLastError("AdaptiveSphSimulation::calculateMergeCriteria: launch velocity criterion");
    }
}
}
