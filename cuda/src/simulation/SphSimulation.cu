#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <vector_types.h>

#include <algorithm>
#include <cstdint>
#include <cuda/simulation/Simulation.cuh>
#include <cuda/simulation/physics/StaticBoundaryDomain.cuh>
#include <glm/common.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/ext/vector_uint3.hpp>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "SphSimulation.cuh"
#include "algorithm/Algorithm.cuh"
#include "algorithm/BoundaryAlgorithm.cuh"
#include "algorithm/WendlandKernel.cuh"
#include "memory/ImportedParticleMemory.cuh"
#include "neighbors/NeighborGrid.cuh"
#include "utils/CudaError.cuh"
#include "utils/DeviceBuffer.cuh"

namespace sph::cuda
{
namespace
{
auto checkedParticleCount(std::size_t particleCount) -> uint32_t
{
    if (particleCount > std::numeric_limits<uint32_t>::max())
    {
        throw std::length_error {"SphSimulation particle count exceeds the supported 32-bit range"};
    }
    return static_cast<uint32_t>(particleCount);
}

template <typename T>
void requireImportedCapacity(const ImportedParticleMemory& memory, std::size_t particleCount, const char* bufferName)
{
    const auto capacity = memory.getMaxDataCount<T>();
    if (particleCount > capacity)
    {
        throw std::length_error {std::string {"SphSimulation "} + bufferName + " requires " +
                                 std::to_string(particleCount) + " elements but its imported buffer holds " +
                                 std::to_string(capacity)};
    }
}
}

SphSimulation::SphSimulation(const Parameters& initialParameters,
                             const std::vector<glm::vec4>& positions,
                             const FluidParticlesDataImportedBuffer& fluidParticleMemory,
                             const BoundaryParticlesDataImportedBuffer& boundaryParticleMemory,
                             const physics::StaticBoundaryDomain& boundaryDomain,
                             uint32_t maxFluidParticleCapacity,
                             uint32_t maxBoundaryParticleCapacity,
                             const ScenarioParameters& scenarioParams,
                             float multiplier)
    : _fluidParticlesData {
          .imported = {toInternalBuffer(fluidParticleMemory)},
          .internal = {.accelerations = thrust::device_vector<glm::vec4>(maxFluidParticleCapacity,
                       glm::vec4 {initialParameters.gravity, 0.F}),
                       .smoothingRadii = thrust::device_vector<float>(maxFluidParticleCapacity),
                       .masses = thrust::device_vector<float>(maxFluidParticleCapacity)}
},
      _boundaryParticlesData {.imported = {toInternalBuffer(boundaryParticleMemory)}, .internal = {}},
      _simulationData {initialParameters}, _boundaryParticleCount {},
      _particleCount {checkedParticleCount(positions.size())}, _boundaryParticleCapacity {maxBoundaryParticleCapacity},
      _grid {initialParameters.domain,
             2 * initialParameters.baseSmoothingRadius* device::constant::wendlandRangeRatio* multiplier,
             maxFluidParticleCapacity,
             maxBoundaryParticleCapacity},
      _scenarioParams {scenarioParams}

{
    if (positions.size() > maxFluidParticleCapacity)
    {
        throw std::length_error {"SphSimulation initial fluid particle count exceeds maxFluidParticleCapacity"};
    }

    requireImportedCapacity<glm::vec4>(_fluidParticlesData.imported.positions,
                                       maxFluidParticleCapacity,
                                       "fluid positions");
    requireImportedCapacity<glm::vec4>(_fluidParticlesData.imported.velocities,
                                       maxFluidParticleCapacity,
                                       "fluid velocities");
    requireImportedCapacity<float>(_fluidParticlesData.imported.densities, maxFluidParticleCapacity, "fluid densities");
    requireImportedCapacity<float>(_fluidParticlesData.imported.radii, maxFluidParticleCapacity, "fluid radii");

    const auto velocitiesVec = std::vector(positions.size(), glm::vec4 {});
    const auto radiusesVec = std::vector(positions.size(), initialParameters.baseParticleRadius);

    if (!positions.empty())
    {
        detail::checkCuda(cudaMemcpy(_fluidParticlesData.imported.positions.getData<glm::vec4>(),
                                     positions.data(),
                                     positions.size() * sizeof(glm::vec4),
                                     cudaMemcpyHostToDevice),
                          "SphSimulation constructor: copy initial fluid positions");

        detail::checkCuda(cudaMemcpy(_fluidParticlesData.imported.velocities.getData<glm::vec4>(),
                                     velocitiesVec.data(),
                                     velocitiesVec.size() * sizeof(glm::vec4),
                                     cudaMemcpyHostToDevice),
                          "SphSimulation constructor: copy initial fluid velocities");

        detail::checkCuda(cudaMemcpy(_fluidParticlesData.imported.radii.getData<float>(),
                                     radiusesVec.data(),
                                     radiusesVec.size() * sizeof(float),
                                     cudaMemcpyHostToDevice),
                          "SphSimulation constructor: copy initial fluid radii");
    }

    thrust::fill_n(_fluidParticlesData.internal.smoothingRadii.begin(),
                   positions.size(),
                   initialParameters.baseSmoothingRadius);

    thrust::fill_n(_fluidParticlesData.internal.masses.begin(), positions.size(), initialParameters.baseParticleMass);

    initializeBoundaryParticles(boundaryDomain);
    _grid.updateBoundary({getBlocksPerGridForBoundaryParticles(), _simulationData.threadsPerBlock},
                         getBoundaryParticles().positions,
                         getBoundaryParticlesCount());
}

auto SphSimulation::toInternalBuffer(const FluidParticlesDataImportedBuffer& memory)
    -> FluidParticlesDataBuffer::Imported
{
    return {.positions = dynamic_cast<const ImportedParticleMemory&>(memory.positions),
            .velocities = dynamic_cast<const ImportedParticleMemory&>(memory.velocities),
            .densities = dynamic_cast<const ImportedParticleMemory&>(memory.densities),
            .radii = dynamic_cast<const ImportedParticleMemory&>(memory.radii)};
}

auto SphSimulation::toInternalBuffer(const BoundaryParticlesDataImportedBuffer& memory)
    -> BoundaryParticlesDataBuffer::Imported
{
    return {
        .positions = dynamic_cast<const ImportedParticleMemory&>(memory.positions),
        .radii = dynamic_cast<const ImportedParticleMemory&>(memory.radii),
        .colors = dynamic_cast<const ImportedParticleMemory&>(memory.colors),
    };
}

void SphSimulation::update(float deltaTime)
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

    halfKickVelocities(deltaTime / 2.F);

    detail::checkCuda(cudaDeviceSynchronize(), "SphSimulation::update: synchronize completed timestep");

    _accumulatedSimulationTime += deltaTime;
}

void SphSimulation::updateDomain(const Parameters::Domain& domain, const physics::StaticBoundaryDomain& boundaryDomain)
{
    detail::checkCuda(cudaDeviceSynchronize(), "SphSimulation::updateDomain: synchronize before domain change");
    validateBoundaryParticleCapacity(boundaryDomain.getParticleCount());
    _simulationData.domain = domain;
    initializeBoundaryParticles(boundaryDomain);
    _grid.updateBoundarySize({getBlocksPerGridForFluidParticles(), _simulationData.threadsPerBlock},
                             {getBlocksPerGridForBoundaryParticles(), _simulationData.threadsPerBlock},
                             domain,
                             getFluidParticles().positions,
                             thrust::raw_pointer_cast(_fluidParticlesData.internal.smoothingRadii.data()),
                             getFluidParticlesCount(),
                             getBoundaryParticles().positions,
                             getBoundaryParticlesCount());
}

auto SphSimulation::getBlocksPerGridForFluidParticles() const -> dim3
{
    return {(_particleCount + _simulationData.threadsPerBlock - 1) / _simulationData.threadsPerBlock};
}

auto SphSimulation::getBlocksPerGridForBoundaryParticles() const -> dim3
{
    return {(_boundaryParticleCount + _simulationData.threadsPerBlock - 1) / _simulationData.threadsPerBlock};
}

auto SphSimulation::getFluidParticles() -> FluidParticlesData
{
    return {
        .positions = _fluidParticlesData.imported.positions
                         .getData<std::remove_pointer_t<decltype(FluidParticlesData::positions)>>(),
        .velocities = _fluidParticlesData.imported.velocities
                          .getData<std::remove_pointer_t<decltype(FluidParticlesData::velocities)>>(),
        .accelerations = thrust::raw_pointer_cast(_fluidParticlesData.internal.accelerations.data()),
        .densities = _fluidParticlesData.imported.densities
                         .getData<std::remove_pointer_t<decltype(FluidParticlesData::densities)>>(),
        .radii =
            _fluidParticlesData.imported.radii.getData<std::remove_pointer_t<decltype(FluidParticlesData::radii)>>(),
        .smoothingRadii = thrust::raw_pointer_cast(_fluidParticlesData.internal.smoothingRadii.data()),
        .masses = thrust::raw_pointer_cast(_fluidParticlesData.internal.masses.data()),
        .particleCount = _particleCount};
}

auto SphSimulation::getBoundaryParticles() -> BoundaryParticlesData
{
    return {
        .positions = _boundaryParticlesData.imported.positions
                         .getData<std::remove_pointer_t<decltype(BoundaryParticlesData::positions)>>(),
        .colors = _boundaryParticlesData.imported.colors
                      .getData<std::remove_pointer_t<decltype(BoundaryParticlesData::colors)>>(),
        .psiValues = thrust::raw_pointer_cast(_boundaryParticlesData.internal.psiValues.data()),
        .viscosityCoefficients = thrust::raw_pointer_cast(_boundaryParticlesData.internal.viscosityCoefficients.data()),
        .radii = _boundaryParticlesData.imported.radii
                     .getData<std::remove_pointer_t<decltype(BoundaryParticlesData::radii)>>(),
        .particleCount = _boundaryParticleCount};
}

void SphSimulation::computeExternalAccelerations()
{
    kernel::computeExternalAccelerations<<<getBlocksPerGridForFluidParticles(), _simulationData.threadsPerBlock>>>(
        getFluidParticles(),
        _simulationData,
        _scenarioParams,
        _scenarioActivated);
    detail::checkCudaLastError("SphSimulation::computeExternalAccelerations: launch kernel");
}

void SphSimulation::computeDensities()
{
    if (_scenarioParams.type == ScenarioType::PoiseuilleFlow)
    {
        kernel::computeDensitiesPeriodic<<<getBlocksPerGridForFluidParticles(), _simulationData.threadsPerBlock>>>(
            getFluidParticles(),
            _grid.toDevice());
        detail::checkCudaLastError("SphSimulation::computeDensities: launch periodic kernel");
    }
    else
    {
        kernel::computeDensities<<<getBlocksPerGridForFluidParticles(), _simulationData.threadsPerBlock>>>(
            getFluidParticles(),
            _grid.toDevice());
        detail::checkCudaLastError("SphSimulation::computeDensities: launch kernel");
    }
}

void SphSimulation::computePressureAccelerations()
{
    if (_scenarioParams.type == ScenarioType::PoiseuilleFlow)
    {
        kernel::computePressureAccelerationsPeriodic<<<getBlocksPerGridForFluidParticles(),
                                                       _simulationData.threadsPerBlock>>>(getFluidParticles(),
                                                                                          _grid.toDevice(),
                                                                                          _simulationData);
        detail::checkCudaLastError("SphSimulation::computePressureAccelerations: launch periodic kernel");
    }
    else
    {
        kernel::computePressureAccelerations<<<getBlocksPerGridForFluidParticles(), _simulationData.threadsPerBlock>>>(
            getFluidParticles(),
            _grid.toDevice(),
            _simulationData);
        detail::checkCudaLastError("SphSimulation::computePressureAccelerations: launch kernel");
    }
}

void SphSimulation::computeViscosityAccelerations()
{
    if (_simulationData.viscosityConstant != 0.F)
    {
        if (_scenarioParams.type == ScenarioType::PoiseuilleFlow)
        {
            kernel::computeViscosityAccelerationsPeriodic<<<getBlocksPerGridForFluidParticles(),
                                                            _simulationData.threadsPerBlock>>>(getFluidParticles(),
                                                                                               _grid.toDevice(),
                                                                                               _simulationData);
            detail::checkCudaLastError("SphSimulation::computeViscosityAccelerations: launch periodic kernel");
        }
        else
        {
            kernel::
                computeViscosityAccelerations<<<getBlocksPerGridForFluidParticles(), _simulationData.threadsPerBlock>>>(
                    getFluidParticles(),
                    _grid.toDevice(),
                    _simulationData);
            detail::checkCudaLastError("SphSimulation::computeViscosityAccelerations: launch kernel");
        }
    }
}

void SphSimulation::computeSurfaceTensionAccelerations()
{
    if (_simulationData.surfaceTensionConstant != 0.F)
    {
        if (_scenarioParams.type == ScenarioType::PoiseuilleFlow)
        {
            kernel::computeSurfaceTensionAccelerationsPeriodic<<<getBlocksPerGridForFluidParticles(),
                                                                 _simulationData.threadsPerBlock>>>(getFluidParticles(),
                                                                                                    _grid.toDevice(),
                                                                                                    _simulationData);
            detail::checkCudaLastError("SphSimulation::computeSurfaceTensionAccelerations: launch periodic kernel");
        }
        else
        {
            kernel::computeSurfaceTensionAccelerations<<<getBlocksPerGridForFluidParticles(),
                                                         _simulationData.threadsPerBlock>>>(getFluidParticles(),
                                                                                            _grid.toDevice(),
                                                                                            _simulationData);
            detail::checkCudaLastError("SphSimulation::computeSurfaceTensionAccelerations: launch kernel");
        }
    }
}

void SphSimulation::halfKickVelocities(float halfDt)
{
    kernel::halfKickVelocities<<<getBlocksPerGridForFluidParticles(), _simulationData.threadsPerBlock>>>(
        getFluidParticles(),
        _simulationData,
        halfDt);
    detail::checkCudaLastError("SphSimulation::halfKickVelocities: launch kernel");
}

void SphSimulation::updatePositions(float dt)
{
    if (_scenarioParams.type == ScenarioType::PoiseuilleFlow)
    {
        kernel::updatePositionsPeriodic<<<getBlocksPerGridForFluidParticles(), _simulationData.threadsPerBlock>>>(
            getFluidParticles(),
            _simulationData.domain,
            dt);
        detail::checkCudaLastError("SphSimulation::updatePositions: launch periodic kernel");
    }
    else
    {
        kernel::updatePositions<<<getBlocksPerGridForFluidParticles(), _simulationData.threadsPerBlock>>>(
            getFluidParticles(),
            _simulationData.domain,
            dt);
        detail::checkCudaLastError("SphSimulation::updatePositions: launch kernel");
    }
}

auto SphSimulation::calculateAverageNeighborCount() -> float
{
    auto totalNeighborCount =
        DeviceBuffer<uint32_t>::fromValue(0, "SphSimulation::calculateAverageNeighborCount: initialize neighbor count");
    kernel::sumAllNeighbors<<<getBlocksPerGridForFluidParticles(), getThreadsPerBlock()>>>(getFluidParticles(),
                                                                                           _grid.toDevice(),
                                                                                           totalNeighborCount.data());
    detail::checkCudaLastError("SphSimulation::calculateAverageNeighborCount: launch kernel");

    return _particleCount > 0 ? static_cast<float>(totalNeighborCount.copyValueToHost(
                                    "SphSimulation::calculateAverageNeighborCount: read neighbor count")) /
                                    static_cast<float>(_particleCount)
                              : 0.F;
}

auto SphSimulation::getDensityInfo(float threshold) -> DensityInfo
{
    auto info = thrust::transform_reduce(
        thrust::device,
        getFluidParticles().densities,
        getFluidParticles().densities + getFluidParticlesCount(),
        [threshold, restDensity = _simulationData.restDensity] __device__(const float density) -> DensityInfo {
            if (!isfinite(density))
            {
                return DensityInfo {.restDensity = restDensity,
                                    .minDensity = std::numeric_limits<float>::max(),
                                    .maxDensity = std::numeric_limits<float>::lowest(),
                                    .averageDensity = 0.0F,
                                    .underDensityCount = 0U,
                                    .normalDensityCount = 0U,
                                    .overDensityCount = 0U};
            }
            const auto ratio = density / restDensity - 1.0F;
            return DensityInfo {.restDensity = restDensity,
                                .minDensity = density,
                                .maxDensity = density,
                                .averageDensity = density,
                                .underDensityCount = (ratio < -threshold) ? 1U : 0U,
                                .normalDensityCount = (ratio >= -threshold && ratio <= threshold) ? 1U : 0U,
                                .overDensityCount = (ratio > threshold) ? 1U : 0U};
        },
        DensityInfo {.restDensity = _simulationData.restDensity,
                     .minDensity = std::numeric_limits<float>::max(),
                     .maxDensity = std::numeric_limits<float>::lowest(),
                     .averageDensity = 0.0f,
                     .underDensityCount = 0,
                     .normalDensityCount = 0,
                     .overDensityCount = 0},
        [restDensity = _simulationData.restDensity] __device__(const DensityInfo& a,
                                                               const DensityInfo& b) -> DensityInfo {
            return DensityInfo {.restDensity = restDensity,
                                .minDensity = std::min(a.minDensity, b.minDensity),
                                .maxDensity = std::max(a.maxDensity, b.maxDensity),
                                .averageDensity = a.averageDensity + b.averageDensity,
                                .underDensityCount = a.underDensityCount + b.underDensityCount,
                                .normalDensityCount = a.normalDensityCount + b.normalDensityCount,
                                .overDensityCount = a.overDensityCount + b.overDensityCount};
        });

    const auto validParticleCount = info.underDensityCount + info.normalDensityCount + info.overDensityCount;
    if (validParticleCount > 0U)
    {
        info.averageDensity /= static_cast<float>(validParticleCount);
    }
    else
    {
        info.averageDensity = 0.0F;
    }
    return info;
}

void SphSimulation::initializeBoundaryParticles(const physics::StaticBoundaryDomain& boundaryDomain)
{
    const auto& particles = boundaryDomain.getParticles();
    const auto particleCount = boundaryDomain.getParticleCount();
    validateBoundaryParticleCapacity(particleCount);

    if (particleCount == 0)
    {
        _boundaryParticleCount = 0;
        _boundaryParticlesData.internal.psiValues.clear();
        _boundaryParticlesData.internal.viscosityCoefficients.clear();
        return;
    }

    std::vector<glm::vec4> positions;
    std::vector<float> psiValues;
    std::vector<glm::vec4> colors;
    std::vector<float> viscosityCoeffs;
    std::vector radii(particleCount, _simulationData.baseParticleRadius);

    positions.reserve(particleCount);
    psiValues.reserve(particleCount);
    colors.reserve(particleCount);
    viscosityCoeffs.reserve(particleCount);

    for (const auto& particle : particles)
    {
        positions.push_back(particle.position);
        psiValues.push_back(particle.psi);
        colors.emplace_back(0.3F, 0.3F, 0.3F, 0.2F);
        viscosityCoeffs.push_back(_simulationData.domain.friction);
    }

    detail::checkCuda(cudaMemcpy(_boundaryParticlesData.imported.positions.getData<glm::vec4>(),
                                 positions.data(),
                                 particleCount * sizeof(glm::vec4),
                                 cudaMemcpyHostToDevice),
                      "SphSimulation::initializeBoundaryParticles: copy positions");

    detail::checkCuda(cudaMemcpy(_boundaryParticlesData.imported.colors.getData<glm::vec4>(),
                                 colors.data(),
                                 particleCount * sizeof(glm::vec4),
                                 cudaMemcpyHostToDevice),
                      "SphSimulation::initializeBoundaryParticles: copy colors");

    detail::checkCuda(cudaMemcpy(_boundaryParticlesData.imported.radii.getData<float>(),
                                 radii.data(),
                                 particleCount * sizeof(float),
                                 cudaMemcpyHostToDevice),
                      "SphSimulation::initializeBoundaryParticles: copy radii");

    _boundaryParticlesData.internal.psiValues = psiValues;
    _boundaryParticlesData.internal.viscosityCoefficients = viscosityCoeffs;
    _boundaryParticleCount = particleCount;
}

void SphSimulation::validateBoundaryParticleCapacity(uint32_t particleCount) const
{
    if (particleCount > _boundaryParticleCapacity)
    {
        throw std::length_error {"SphSimulation boundary particle count " + std::to_string(particleCount) +
                                 " exceeds allocated capacity " + std::to_string(_boundaryParticleCapacity)};
    }

    requireImportedCapacity<glm::vec4>(_boundaryParticlesData.imported.positions, particleCount, "boundary positions");
    requireImportedCapacity<float>(_boundaryParticlesData.imported.radii, particleCount, "boundary radii");
    requireImportedCapacity<glm::vec4>(_boundaryParticlesData.imported.colors, particleCount, "boundary colors");
}

void SphSimulation::computeBoundaryDensityContribution()
{
    if (_scenarioParams.type == ScenarioType::PoiseuilleFlow)
    {
        kernel::
            computeBoundaryDensityContributionPeriodic<<<getBlocksPerGridForFluidParticles(), getThreadsPerBlock()>>>(
                getFluidParticles(),
                getBoundaryParticles(),
                _grid.toDevice());
        detail::checkCudaLastError("SphSimulation::computeBoundaryDensityContribution: launch periodic kernel");
    }
    else
    {
        kernel::computeBoundaryDensityContribution<<<getBlocksPerGridForFluidParticles(), getThreadsPerBlock()>>>(
            getFluidParticles(),
            getBoundaryParticles(),
            _grid.toDevice());
        detail::checkCudaLastError("SphSimulation::computeBoundaryDensityContribution: launch kernel");
    }
}

void SphSimulation::computeBoundaryForces()
{
    if (_scenarioParams.type == ScenarioType::PoiseuilleFlow)
    {
        kernel::
            computeBoundaryPressureAccelerationPeriodic<<<getBlocksPerGridForFluidParticles(), getThreadsPerBlock()>>>(
                getFluidParticles(),
                getBoundaryParticles(),
                _grid.toDevice(),
                _simulationData);
        detail::checkCudaLastError("SphSimulation::computeBoundaryForces: launch periodic pressure kernel");

        kernel::
            computeBoundaryFrictionAccelerationPeriodic<<<getBlocksPerGridForFluidParticles(), getThreadsPerBlock()>>>(
                getFluidParticles(),
                getBoundaryParticles(),
                _grid.toDevice(),
                _simulationData,
                0.0F);
        detail::checkCudaLastError("SphSimulation::computeBoundaryForces: launch periodic friction kernel");
    }
    else
    {
        kernel::computeBoundaryPressureAcceleration<<<getBlocksPerGridForFluidParticles(), getThreadsPerBlock()>>>(
            getFluidParticles(),
            getBoundaryParticles(),
            _grid.toDevice(),
            _simulationData);
        detail::checkCudaLastError("SphSimulation::computeBoundaryForces: launch pressure kernel");

        const auto lidVelocity = (_scenarioParams.type == ScenarioType::LidDrivenCavity &&
                                  _scenarioParams.lidDrivenCavity.has_value() && _scenarioActivated)
                                     ? _scenarioParams.lidDrivenCavity->lidVelocity
                                     : 0.0F;

        kernel::computeBoundaryFrictionAcceleration<<<getBlocksPerGridForFluidParticles(), getThreadsPerBlock()>>>(
            getFluidParticles(),
            getBoundaryParticles(),
            _grid.toDevice(),
            _simulationData,
            lidVelocity);
        detail::checkCudaLastError("SphSimulation::computeBoundaryForces: launch friction kernel");
    }
}

void SphSimulation::updateGrid()
{
    _grid.updateFluid({getBlocksPerGridForFluidParticles(), _simulationData.threadsPerBlock},
                      getFluidParticles().positions,
                      thrust::raw_pointer_cast(_fluidParticlesData.internal.smoothingRadii.data()),
                      getFluidParticlesCount());
}

void SphSimulation::enableAdaptiveRefinement() { }

auto SphSimulation::getCurrentSimulationTime() const -> float
{
    return _accumulatedSimulationTime;
}

}
