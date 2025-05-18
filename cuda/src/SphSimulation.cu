#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <vector_types.h>

#include <cstddef>
#include <cstdint>
#include <cuda/Simulation.cuh>
#include <glm/common.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/ext/vector_uint3.hpp>
#include <span>
#include <type_traits>
#include <vector>

#include "Algorithm.cuh"
#include "ImportedParticleMemory.cuh"
#include "SphSimulation.cuh"
#include "common/Utils.cuh"

namespace sph::cuda
{

SphSimulation::SphSimulation(const Parameters& initialParameters,
                             const std::vector<glm::vec4>& positions,
                             const ParticlesDataBuffer& memory,
                             uint32_t maxParticleCapacity)
    : _particleBuffer {toInternalBuffer(memory)},
      _simulationData {initialParameters},
      _state {.grid = createGrid(initialParameters, maxParticleCapacity)},
      _particleCount {static_cast<uint32_t>(positions.size())},
      _particleCapacity {maxParticleCapacity}
{
    const auto velocitiesVec = std::vector(positions.size(), glm::vec4 {});
    const auto radiusesVec = std::vector(positions.size(), initialParameters.baseParticleRadius);
    const auto smoothingRadiusesVec = std::vector(positions.size(), initialParameters.baseSmoothingRadius);
    const auto massesVec = std::vector(positions.size(), initialParameters.baseParticleMass);

    cudaMemcpy(_particleBuffer.positions.getData<glm::vec4>(),
               positions.data(),
               positions.size() * sizeof(glm::vec4),
               cudaMemcpyHostToDevice);

    cudaMemcpy(_particleBuffer.velocities.getData<glm::vec4>(),
               velocitiesVec.data(),
               velocitiesVec.size() * sizeof(glm::vec4),
               cudaMemcpyHostToDevice);

    cudaMemcpy(_particleBuffer.predictedPositions.getData<glm::vec4>(),
               positions.data(),
               positions.size() * sizeof(glm::vec4),
               cudaMemcpyHostToDevice);

    cudaMemcpy(_particleBuffer.radiuses.getData<float>(),
               radiusesVec.data(),
               radiusesVec.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemcpy(_particleBuffer.smoothingRadiuses.getData<float>(),
               smoothingRadiusesVec.data(),
               smoothingRadiusesVec.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemcpy(_particleBuffer.masses.getData<float>(),
               massesVec.data(),
               massesVec.size() * sizeof(float),
               cudaMemcpyHostToDevice);
}

SphSimulation::~SphSimulation()
{
    cudaFree(_state.grid.particleGridIndices.data());
    cudaFree(_state.grid.particleArrayIndices.data());
    cudaFree(_state.grid.cellStartIndices.data());
    cudaFree(_state.grid.cellEndIndices.data());
}

auto SphSimulation::toInternalBuffer(const ParticlesDataBuffer& memory) -> ParticlesInternalDataBuffer
{
    return {.positions = dynamic_cast<const ImportedParticleMemory&>(memory.positions),
            .predictedPositions = dynamic_cast<const ImportedParticleMemory&>(memory.predictedPositions),
            .velocities = dynamic_cast<const ImportedParticleMemory&>(memory.velocities),
            .densities = dynamic_cast<const ImportedParticleMemory&>(memory.densities),
            .nearDensities = dynamic_cast<const ImportedParticleMemory&>(memory.nearDensities),
            .pressures = dynamic_cast<const ImportedParticleMemory&>(memory.pressures),
            .radiuses = dynamic_cast<const ImportedParticleMemory&>(memory.radiuses),
            .smoothingRadiuses = dynamic_cast<const ImportedParticleMemory&>(memory.smoothingRadiuses),
            .masses = dynamic_cast<const ImportedParticleMemory&>(memory.masses),
            .densityDeviations = dynamic_cast<const ImportedParticleMemory&>(memory.densityDeviations)};
}

void SphSimulation::update(float deltaTime)
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
    handleCollisions();

    cudaDeviceSynchronize();
}

auto SphSimulation::createGrid(const Parameters& data, size_t particleCapacity) -> Grid
{
    int32_t* particleIndices {};
    int32_t* particleArrayIndices {};
    int32_t* cellStartIndices {};
    int32_t* cellEndIndices {};

    const auto gridCellWidth = 2 * data.baseSmoothingRadius;
    const auto gridCellCount = glm::uvec3 {glm::ceil((data.domain.max - data.domain.min) / gridCellWidth)};

    cudaMalloc(&particleIndices, particleCapacity * sizeof(int32_t));
    cudaMalloc(&particleArrayIndices, particleCapacity * sizeof(int32_t));
    cudaMalloc(&cellStartIndices,
               static_cast<size_t>(gridCellCount.x * gridCellCount.y * gridCellCount.z) * sizeof(int32_t));
    cudaMalloc(&cellEndIndices,
               static_cast<size_t>(gridCellCount.x * gridCellCount.y * gridCellCount.z) * sizeof(int32_t));

    return Grid {
        .gridSize = gridCellCount,
        .cellSize = glm::vec3 {gridCellWidth},
        .cellStartIndices =
            std::span {cellStartIndices, static_cast<size_t>(gridCellCount.x * gridCellCount.y * gridCellCount.z)},
        .cellEndIndices =
            std::span {cellEndIndices, static_cast<size_t>(gridCellCount.x * gridCellCount.y * gridCellCount.z)},
        .particleGridIndices = std::span {particleIndices, particleCapacity},
        .particleArrayIndices = std::span {particleArrayIndices, particleCapacity}
    };
}

auto SphSimulation::getBlocksPerGridForParticles() const -> dim3
{
    return {(_particleCount + _simulationData.threadsPerBlock - 1) / _simulationData.threadsPerBlock};
}

auto SphSimulation::getBlocksPerGridForGrid() const -> dim3
{
    return {(_state.grid.gridSize.x * _state.grid.gridSize.y * _state.grid.gridSize.z +
             _simulationData.threadsPerBlock - 1) /
            _simulationData.threadsPerBlock};
}

auto SphSimulation::getParticles() const -> ParticlesData
{
    return {
        .positions = _particleBuffer.positions.getData<std::remove_pointer_t<decltype(ParticlesData::positions)>>(),
        .predictedPositions = _particleBuffer.predictedPositions
                                  .getData<std::remove_pointer_t<decltype(ParticlesData::predictedPositions)>>(),
        .velocities = _particleBuffer.velocities.getData<std::remove_pointer_t<decltype(ParticlesData::velocities)>>(),
        .densities = _particleBuffer.densities.getData<std::remove_pointer_t<decltype(ParticlesData::densities)>>(),
        .nearDensities =
            _particleBuffer.nearDensities.getData<std::remove_pointer_t<decltype(ParticlesData::nearDensities)>>(),
        .pressures = _particleBuffer.pressures.getData<std::remove_pointer_t<decltype(ParticlesData::pressures)>>(),
        .radiuses = _particleBuffer.radiuses.getData<std::remove_pointer_t<decltype(ParticlesData::radiuses)>>(),
        .smoothingRadiuses = _particleBuffer.smoothingRadiuses
                                 .getData<std::remove_pointer_t<decltype(ParticlesData::smoothingRadiuses)>>(),
        .masses = _particleBuffer.masses.getData<std::remove_pointer_t<decltype(ParticlesData::masses)>>(),
        .densityDeviations = _particleBuffer.densityDeviations
                                 .getData<std::remove_pointer_t<decltype(ParticlesData::densityDeviations)>>(),
        .particleCount = _particleCount};
}

void SphSimulation::computeExternalForces(float deltaTime) const
{
    kernel::computeExternalForces<<<getBlocksPerGridForParticles(), _simulationData.threadsPerBlock>>>(getParticles(),
                                                                                                       _simulationData,
                                                                                                       deltaTime);
}

void SphSimulation::resetGrid() const
{
    kernel::resetGrid<<<getBlocksPerGridForGrid(), _simulationData.threadsPerBlock>>>(_state.grid);
}

void SphSimulation::assignParticlesToCells() const
{
    kernel::assignParticlesToCells<<<getBlocksPerGridForParticles(), _simulationData.threadsPerBlock>>>(
        getParticles(),
        _state,
        _simulationData);
}

void SphSimulation::sortParticles() const
{
    thrust::sort_by_key(thrust::device,
                        _state.grid.particleGridIndices.data(),
                        _state.grid.particleGridIndices.data() + getParticlesCount(),
                        _state.grid.particleArrayIndices.data());
}

void SphSimulation::calculateCellStartAndEndIndices() const
{
    kernel::calculateCellStartAndEndIndices<<<getBlocksPerGridForParticles(), _simulationData.threadsPerBlock>>>(
        _state.grid,
        getParticlesCount());
}

void SphSimulation::computeDensities() const
{
    kernel::computeDensities<<<getBlocksPerGridForParticles(), _simulationData.threadsPerBlock>>>(getParticles(),
                                                                                                  _state,
                                                                                                  _simulationData);
}

void SphSimulation::computePressureForce(float deltaTime) const
{
    kernel::computePressureForce<<<getBlocksPerGridForParticles(), _simulationData.threadsPerBlock>>>(getParticles(),
                                                                                                      _state,
                                                                                                      _simulationData,
                                                                                                      deltaTime);
}

void SphSimulation::computeViscosityForce(float deltaTime) const
{
    kernel::computeViscosityForce<<<getBlocksPerGridForParticles(), _simulationData.threadsPerBlock>>>(getParticles(),
                                                                                                       _state,
                                                                                                       _simulationData,
                                                                                                       deltaTime);
}

void SphSimulation::integrateMotion(float deltaTime) const
{
    kernel::integrateMotion<<<getBlocksPerGridForParticles(), _simulationData.threadsPerBlock>>>(getParticles(),
                                                                                                 _simulationData,
                                                                                                 deltaTime);
}

void SphSimulation::handleCollisions() const
{
    kernel::handleCollisions<<<getBlocksPerGridForParticles(), _simulationData.threadsPerBlock>>>(getParticles(),
                                                                                                  _simulationData);
}

auto SphSimulation::calculateAverageNeighborCount() const -> float
{
    std::vector<uint32_t> neighborCounts(_particleCount, 0);
    uint32_t* d_neighborCounts = nullptr;
    cudaMalloc(&d_neighborCounts, _particleCount * sizeof(uint32_t));
    cudaMemcpy(d_neighborCounts, neighborCounts.data(), _particleCount * sizeof(uint32_t), cudaMemcpyHostToDevice);
    kernel::countNeighbors<<<getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(getParticles(),
                                                                                     _state,
                                                                                     _simulationData,
                                                                                     d_neighborCounts);
    cudaMemcpy(neighborCounts.data(), d_neighborCounts, _particleCount * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_neighborCounts);

    uint32_t totalNeighbors = 0;
    for (uint32_t i = 0; i < _particleCount; i++)
    {
        totalNeighbors += neighborCounts[i];
    }

    return _particleCount > 0 ? static_cast<float>(totalNeighbors) / static_cast<float>(_particleCount) : 0.F;
}

std::vector<float> SphSimulation::updateDensityDeviations() const
{
    if (_particleCount == 0)
    {
        return {};
    }

    kernel::calculateDensityDeviations<<<getBlocksPerGridForParticles(), _simulationData.threadsPerBlock>>>(
        getParticles(),
        _simulationData.restDensity);

    return fromGpu(getParticles().densityDeviations, getParticlesCount());
}

void SphSimulation::setParticleVelocity(uint32_t particleIndex, const glm::vec4& velocity)
{
    if (particleIndex < _particleCount)
    {
        cudaMemcpy(_particleBuffer.velocities.getData<glm::vec4>() + particleIndex,
                   &velocity,
                   sizeof(glm::vec4),
                   cudaMemcpyHostToDevice);
    }
}

}
