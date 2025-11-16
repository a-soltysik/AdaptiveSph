#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <vector_types.h>

#include <cuda/Simulation.cuh>
#include <glm/common.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/ext/vector_uint3.hpp>
#include <span>
#include <type_traits>
#include <vector>

#include "SphSimulation.cuh"
#include "algorithm/Algorithm.cuh"
#include "algorithm/BoundaryAlgorithm.cuh"
#include "algorithm/NeighborGrid.cuh"
#include "algorithm/kernels/Kernel.cuh"
#include "cuda/physics/StaticBoundaryDomain.cuh"
#include "memory/ImportedParticleMemory.cuh"
#include "utils/DeviceValue.cuh"

namespace sph::cuda
{

SphSimulation::SphSimulation(const Parameters& initialParameters,
                             const std::vector<glm::vec4>& positions,
                             const FluidParticlesDataImportedBuffer& fluidParticleMemory,
                             const BoundaryParticlesDataImportedBuffer& boundaryParticleMemory,
                             const physics::StaticBoundaryDomain& boundaryDomain,
                             uint32_t maxParticleCapacity)
    : _fluidParticlesData {
          .imported = {toInternalBuffer(fluidParticleMemory)},
          .internal = {.accelerations = thrust::device_vector<glm::vec4>(positions.size(),
                       glm::vec4 {initialParameters.gravity, 0.F}),
                       .smoothingRadii = thrust::device_vector<float>(positions.size()),
                       .masses = thrust::device_vector<float>(positions.size())}
},
      _boundaryParticlesData {.imported = {toInternalBuffer(boundaryParticleMemory)}, .internal = {}},
      _simulationData {initialParameters}, _boundaryParticleCount {boundaryDomain.getParticleCount()},
      _particleCount {static_cast<uint32_t>(positions.size())}, _particleCapacity {maxParticleCapacity},
      _grid {initialParameters.domain,
             2 * initialParameters.baseSmoothingRadius* device::constant::wendlandRangeRatio,
             _particleCapacity,
             _boundaryParticleCount}

{
    const auto velocitiesVec = std::vector(positions.size(), glm::vec4 {});
    const auto radiusesVec = std::vector(positions.size(), initialParameters.baseParticleRadius);

    cudaMemcpy(_fluidParticlesData.imported.positions.getData<glm::vec4>(),
               positions.data(),
               positions.size() * sizeof(glm::vec4),
               cudaMemcpyHostToDevice);

    cudaMemcpy(_fluidParticlesData.imported.velocities.getData<glm::vec4>(),
               velocitiesVec.data(),
               velocitiesVec.size() * sizeof(glm::vec4),
               cudaMemcpyHostToDevice);

    cudaMemcpy(_fluidParticlesData.imported.radii.getData<float>(),
               radiusesVec.data(),
               radiusesVec.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    thrust::fill(_fluidParticlesData.internal.smoothingRadii.begin(),
                 _fluidParticlesData.internal.smoothingRadii.end(),
                 initialParameters.baseSmoothingRadius);

    thrust::fill(_fluidParticlesData.internal.masses.begin(),
                 _fluidParticlesData.internal.masses.end(),
                 initialParameters.baseParticleMass);

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

    _grid.updateFluid({getBlocksPerGridForFluidParticles(), _simulationData.threadsPerBlock},
                      getFluidParticles().positions,
                      getFluidParticlesCount());
    computeDensities();
    computeBoundaryDensityContribution();

    computeExternalAccelerations();
    computePressureAccelerations();
    computeViscosityAccelerations();
    computeSurfaceTensionAccelerations();

    computeBoundaryForces();

    halfKickVelocities(deltaTime / 2.F);

    cudaDeviceSynchronize();
}

void SphSimulation::updateDomain(const Parameters::Domain& domain, const physics::StaticBoundaryDomain& boundaryDomain)
{
    initializeBoundaryParticles(boundaryDomain);
    _grid.updateBoundarySize({getBlocksPerGridForFluidParticles(), _simulationData.threadsPerBlock},
                             domain,
                             getFluidParticles().positions,
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
        _simulationData);
}

void SphSimulation::computeDensities()
{
    kernel::computeDensities<<<getBlocksPerGridForFluidParticles(), _simulationData.threadsPerBlock>>>(
        getFluidParticles(),
        _grid.toDevice(),
        _simulationData);
}

void SphSimulation::computePressureAccelerations()
{
    kernel::computePressureAccelerations<<<getBlocksPerGridForFluidParticles(), _simulationData.threadsPerBlock>>>(
        getFluidParticles(),
        _grid.toDevice(),
        _simulationData);
}

void SphSimulation::computeViscosityAccelerations()
{
    kernel::computeViscosityAccelerations<<<getBlocksPerGridForFluidParticles(), _simulationData.threadsPerBlock>>>(
        getFluidParticles(),
        _grid.toDevice(),
        _simulationData);
}

void SphSimulation::computeSurfaceTensionAccelerations()
{
    kernel::
        computeSurfaceTensionAccelerations<<<getBlocksPerGridForFluidParticles(), _simulationData.threadsPerBlock>>>(
            getFluidParticles(),
            _grid.toDevice(),
            _simulationData);
}

void SphSimulation::halfKickVelocities(float halfDt)
{
    kernel::halfKickVelocities<<<getBlocksPerGridForFluidParticles(), _simulationData.threadsPerBlock>>>(
        getFluidParticles(),
        _simulationData,
        halfDt);
}

void SphSimulation::updatePositions(float dt)
{
    kernel::updatePositions<<<getBlocksPerGridForFluidParticles(), _simulationData.threadsPerBlock>>>(
        getFluidParticles(),
        dt);
}

auto SphSimulation::calculateAverageNeighborCount() -> float
{
    auto totalNeighborCount = DeviceValue<uint32_t>::fromHost(0);
    kernel::sumAllNeighbors<<<getBlocksPerGridForFluidParticles(), getThreadsPerBlock()>>>(
        getFluidParticles(),
        _grid.toDevice(),
        totalNeighborCount.getDevicePtr());

    return _particleCount > 0 ? static_cast<float>(totalNeighborCount.toHost()) / static_cast<float>(_particleCount)
                              : 0.F;
}

auto SphSimulation::getDensityInfo(float threshold) -> DensityInfo
{
    auto info = thrust::transform_reduce(
        thrust::device,
        getFluidParticles().densities,
        getFluidParticles().densities + getFluidParticlesCount(),
        [threshold, restDensity = _simulationData.restDensity] __device__(const float density) -> DensityInfo {
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

    info.averageDensity /= static_cast<float>(getFluidParticlesCount());
    return info;
}

void SphSimulation::initializeBoundaryParticles(const physics::StaticBoundaryDomain& boundaryDomain)
{
    const auto& particles = boundaryDomain.getParticles();
    _boundaryParticleCount = boundaryDomain.getParticleCount();

    if (_boundaryParticleCount == 0)
    {
        return;
    }

    std::vector<glm::vec4> positions;
    std::vector<float> psiValues;
    std::vector<glm::vec4> colors;
    std::vector<float> viscosityCoeffs;
    std::vector radii(_boundaryParticleCount, _simulationData.baseParticleRadius);

    positions.reserve(_boundaryParticleCount);
    psiValues.reserve(_boundaryParticleCount);
    colors.reserve(_boundaryParticleCount);
    viscosityCoeffs.reserve(_boundaryParticleCount);

    for (const auto& particle : particles)
    {
        positions.push_back(particle.position);
        psiValues.push_back(particle.psi);
        colors.emplace_back(0.3F, 0.3F, 0.3F, 0.2F);
        viscosityCoeffs.push_back(_simulationData.domain.friction);
    }

    cudaMemcpy(_boundaryParticlesData.imported.positions.getData<glm::vec4>(),
               positions.data(),
               _boundaryParticleCount * sizeof(glm::vec4),
               cudaMemcpyHostToDevice);

    cudaMemcpy(_boundaryParticlesData.imported.colors.getData<glm::vec4>(),
               colors.data(),
               _boundaryParticleCount * sizeof(glm::vec4),
               cudaMemcpyHostToDevice);

    cudaMemcpy(_boundaryParticlesData.imported.radii.getData<float>(),
               radii.data(),
               _boundaryParticleCount * sizeof(float),
               cudaMemcpyHostToDevice);

    _boundaryParticlesData.internal.psiValues = psiValues;
    _boundaryParticlesData.internal.viscosityCoefficients = viscosityCoeffs;
}

void SphSimulation::computeBoundaryDensityContribution()
{
    kernel::computeBoundaryDensityContribution<<<getBlocksPerGridForFluidParticles(), getThreadsPerBlock()>>>(
        getFluidParticles(),
        getBoundaryParticles(),
        _grid.toDevice());
}

void SphSimulation::computeBoundaryForces()
{
    kernel::computeBoundaryPressureAcceleration<<<getBlocksPerGridForFluidParticles(), getThreadsPerBlock()>>>(
        getFluidParticles(),
        getBoundaryParticles(),
        _grid.toDevice(),
        _simulationData);

    kernel::computeBoundaryFrictionAcceleration<<<getBlocksPerGridForFluidParticles(), getThreadsPerBlock()>>>(
        getFluidParticles(),
        getBoundaryParticles(),
        _grid.toDevice(),
        _simulationData);
}

}
