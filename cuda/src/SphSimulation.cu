#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <vector_types.h>

#include <cstddef>
#include <cstdint>
#include <glm/common.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_uint3.hpp>
#include <memory>
#include <thrust/detail/device_ptr.inl>
#include <thrust/detail/sort.inl>
#include <vector>

#include "Algorithm.cuh"
#include "ImportedParticleMemory.cuh"
#include "Span.cuh"
#include "SphSimulation.cuh"
#include "cuda/ImportedMemory.cuh"
#include "cuda/Simulation.cuh"

namespace sph::cuda
{

SphSimulation::SphSimulation(const Parameters& initialParameters,
                             const std::vector<glm::vec3>& positions,
                             const ImportedMemory& memory)
    : _particleBuffer {dynamic_cast<const ImportedParticleMemory&>(memory)},
      _simulationData {initialParameters},
      _state {.grid = createGrid(initialParameters, positions.size())},
      _particleCount {positions.size()}
{
    std::vector<ParticleData> particles(positions.size());
    for (size_t i = 0; i < positions.size(); i++)
    {
        particles[i] = ParticleData {};
        particles[i].position = positions[i];
        particles[i].mass = initialParameters.particleMass;
        particles[i].radius = initialParameters.particleRadius;
    }

    cudaMemcpy(_particleBuffer.getParticles(),
               particles.data(),
               particles.size() * sizeof(ParticleData),
               cudaMemcpyHostToDevice);
}

SphSimulation::~SphSimulation()
{
    cudaFree(_state.grid.particleGridIndices.data);
    cudaFree(_state.grid.particleArrayIndices.data);
    cudaFree(_state.grid.cellStartIndices.data);
    cudaFree(_state.grid.cellEndIndices.data);
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
                      const std::vector<glm::vec3>& positions,
                      const ImportedMemory& memory) -> std::unique_ptr<Simulation>
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
    return {static_cast<unsigned int>((_particleCount + _simulationData.threadsPerBlock - 1) /
                                      _simulationData.threadsPerBlock)};
}

auto SphSimulation::getBlocksPerGridForGrid() const -> dim3
{
    return {(_state.grid.gridSize.x * _state.grid.gridSize.y * _state.grid.gridSize.z +
             _simulationData.threadsPerBlock - 1) /
            _simulationData.threadsPerBlock};
}

auto SphSimulation::getParticles() const -> Span<ParticleData>
{
    return {.data = _particleBuffer.getParticles(), .size = _particleCount};
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

}
