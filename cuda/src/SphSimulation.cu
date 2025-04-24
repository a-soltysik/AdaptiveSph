#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <thrust/execution_policy.h>
#include <vector_types.h>

#include <cstddef>
#include <cstdint>
#include <cuda/Simulation.cuh>
#include <glm/common.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_uint3.hpp>
#include <memory>
#include <thrust/detail/device_ptr.inl>
#include <thrust/detail/sort.inl>
#include <type_traits>
#include <vector>

#include "Algorithm.cuh"
#include "ImportedParticleMemory.cuh"
#include "Span.cuh"
#include "SphSimulation.cuh"
#include "glm/ext/scalar_constants.hpp"

namespace sph::cuda
{

SphSimulation::SphSimulation(const Parameters& initialParameters,
                             const std::vector<glm::vec4>& positions,
                             const ParticlesDataBuffer& memory)
    : _particleBuffer {toInternalBuffer(memory)},
      _simulationData {initialParameters},
      _state {.grid = createGrid(initialParameters, positions.size())},
      _particleCount {static_cast<uint32_t>(positions.size())}
{
    const auto velocitiesVec = std::vector(positions.size(), glm::vec4(0.0f));
    const auto radiusesVec = std::vector(positions.size(), initialParameters.particleRadius);
    auto massesVec = std::vector(
        positions.size(),
        getParticleMass(initialParameters.domain.getVolume(), initialParameters.restDensity, _particleCount));

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

    cudaMemcpy(_particleBuffer.masses.getData<float>(),
               massesVec.data(),
               massesVec.size() * sizeof(float),
               cudaMemcpyHostToDevice);
}

SphSimulation::~SphSimulation()
{
    cudaFree(_state.grid.particleGridIndices.data);
    cudaFree(_state.grid.particleArrayIndices.data);
    cudaFree(_state.grid.cellStartIndices.data);
    cudaFree(_state.grid.cellEndIndices.data);
}

auto SphSimulation::toInternalBuffer(const ParticlesDataBuffer& memory) -> ParticlesInternalDataBuffer
{
    return {.positions = dynamic_cast<const ImportedParticleMemory&>(memory.positions),
            .predictedPositions = dynamic_cast<const ImportedParticleMemory&>(memory.predictedPositions),
            .velocities = dynamic_cast<const ImportedParticleMemory&>(memory.velocities),
            .forces = dynamic_cast<const ImportedParticleMemory&>(memory.forces),
            .densities = dynamic_cast<const ImportedParticleMemory&>(memory.densities),
            .nearDensities = dynamic_cast<const ImportedParticleMemory&>(memory.nearDensities),
            .pressures = dynamic_cast<const ImportedParticleMemory&>(memory.pressures),
            .radiuses = dynamic_cast<const ImportedParticleMemory&>(memory.radiuses),
            .masses = dynamic_cast<const ImportedParticleMemory&>(memory.masses)};
}

void SphSimulation::update(const Parameters& parameters, float deltaTime)
{
    _simulationData = parameters;

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

auto createSimulation(const Simulation::Parameters& data,
                      const std::vector<glm::vec4>& positions,
                      const ParticlesDataBuffer& memory) -> std::unique_ptr<Simulation>
{
    return std::make_unique<SphSimulation>(data, positions, memory);
}

auto SphSimulation::createGrid(const Parameters& data, size_t particleCount) -> Grid
{
    int32_t* particleIndices {};
    int32_t* particleArrayIndices {};
    int32_t* cellStartIndices {};
    int32_t* cellEndIndices {};

    const auto gridCellWidth = 2 * data.smoothingRadius;
    const auto gridCellCount = glm::uvec3 {glm::ceil((data.domain.max - data.domain.min) / gridCellWidth)};

    cudaMalloc(reinterpret_cast<void**>(&particleIndices), particleCount * sizeof(int32_t));
    cudaMalloc(reinterpret_cast<void**>(&particleArrayIndices), particleCount * sizeof(int32_t));
    cudaMalloc(reinterpret_cast<void**>(&cellStartIndices),
               gridCellCount.x * gridCellCount.y * gridCellCount.z * sizeof(int32_t));
    cudaMalloc(reinterpret_cast<void**>(&cellEndIndices),
               gridCellCount.x * gridCellCount.y * gridCellCount.z * sizeof(int32_t));

    return Grid {
        .gridSize = gridCellCount,
        .cellSize = glm::vec3 {gridCellWidth},
        .cellStartIndices =
            Span {.data = cellStartIndices, .size = gridCellCount.x * gridCellCount.y * gridCellCount.z},
        .cellEndIndices = Span {.data = cellEndIndices, .size = gridCellCount.x * gridCellCount.y * gridCellCount.z},
        .particleGridIndices = Span {.data = particleIndices, .size = particleCount},
        .particleArrayIndices = Span {.data = particleArrayIndices, .size = particleCount}
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
        .forces = _particleBuffer.forces.getData<std::remove_pointer_t<decltype(ParticlesData::forces)>>(),
        .densities = _particleBuffer.densities.getData<std::remove_pointer_t<decltype(ParticlesData::densities)>>(),
        .nearDensities =
            _particleBuffer.nearDensities.getData<std::remove_pointer_t<decltype(ParticlesData::nearDensities)>>(),
        .pressures = _particleBuffer.pressures.getData<std::remove_pointer_t<decltype(ParticlesData::pressures)>>(),
        .radiuses = _particleBuffer.radiuses.getData<std::remove_pointer_t<decltype(ParticlesData::radiuses)>>(),
        .masses = _particleBuffer.masses.getData<std::remove_pointer_t<decltype(ParticlesData::masses)>>(),
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
    thrust::sort_by_key(
        thrust::device,
        thrust::device_pointer_cast(_state.grid.particleGridIndices.data),
        thrust::device_pointer_cast(_state.grid.particleGridIndices.data + _state.grid.particleGridIndices.size),
        thrust::device_pointer_cast(_state.grid.particleArrayIndices.data));
}

void SphSimulation::calculateCellStartAndEndIndices() const
{
    kernel::calculateCellStartAndEndIndices<<<getBlocksPerGridForParticles(), _simulationData.threadsPerBlock>>>(
        _state.grid);
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

auto SphSimulation::getParticleMass(float domainVolume, float restDensity, uint32_t particlesCount) -> float
{
    return domainVolume * restDensity / particlesCount;
}

}
