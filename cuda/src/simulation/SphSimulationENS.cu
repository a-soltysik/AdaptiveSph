#include "SphSImulationENS.cuh"
#include "SphSimulation.cuh"
#include "algorithm/AlgorithmENS.cuh"
#include "cuda/Simulation.cuh"
#include "glm/vec4.hpp"

namespace sph::cuda
{

SphSimulationENS::SphSimulationENS(const Parameters& initialParameters,
                                   const std::vector<glm::vec4>& positions,
                                   const ParticlesDataBuffer& memory,
                                   uint32_t maxParticleCapacity)
    : SphSimulation(initialParameters, positions, memory, maxParticleCapacity)
{
    allocateEnsMemory(maxParticleCapacity);
    calculateOptimalConfiguration();
}

SphSimulationENS::~SphSimulationENS()
{
    freeEnsMemory();
}

void SphSimulationENS::allocateEnsMemory(uint32_t maxParticleCapacity)
{
    cudaMalloc(&_particleRectangles, maxParticleCapacity * sizeof(Rectangle3D));
}

void SphSimulationENS::freeEnsMemory()
{
    if (_particleRectangles != nullptr)
    {
        cudaFree(_particleRectangles);
        _particleRectangles = nullptr;
    }
}

void SphSimulationENS::calculateOptimalConfiguration()
{
    const auto threadsPerBlock = getThreadsPerBlock();
    const auto availableSharedMemory = getParameters().sharedMemorySizePerBlock;

    _particlesPerBatch = calculateParticlesPerBatch(threadsPerBlock, availableSharedMemory);
    _sharedMemorySize = availableSharedMemory;
}

void SphSimulationENS::update(float deltaTime)
{
    computeExternalForces(deltaTime);
    resetGrid();
    assignParticlesToCells();
    sortParticles();
    calculateCellStartAndEndIndices();
    computeParticleRectangles();
    computeDensitiesENS();
    //computePressureForceENS(deltaTime);
    //computeViscosityForceENS(deltaTime);
    integrateMotion(deltaTime);
    handleCollisions();

    cudaDeviceSynchronize();
}

void SphSimulationENS::computeParticleRectangles() const
{
    kernel::computeParticleRectangles<<<getBlocksPerGridForParticles(), getThreadsPerBlock()>>>(getParticles(),
                                                                                                _particleRectangles,
                                                                                                getGrid(),
                                                                                                getParameters());
}

void SphSimulationENS::computeDensitiesENS() const
{
    kernel::computeDensitiesENS<<<getBlocksPerGridForParticles(), getThreadsPerBlock(), _sharedMemorySize>>>(
        getParticles(),
        _particleRectangles,
        getGrid(),
        getParameters(),
        _particlesPerBatch);
}

void SphSimulationENS::computePressureForceENS(float deltaTime) const
{
    kernel::computePressureForceENS<<<getBlocksPerGridForParticles(), getThreadsPerBlock(), _sharedMemorySize>>>(
        getParticles(),
        _particleRectangles,
        getGrid(),
        getParameters(),
        deltaTime,
        _particlesPerBatch);
}

void SphSimulationENS::computeViscosityForceENS(float deltaTime) const
{
    kernel::computeViscosityForceENS<<<getBlocksPerGridForParticles(), getThreadsPerBlock(), _sharedMemorySize>>>(
        getParticles(),
        _particleRectangles,
        getGrid(),
        getParameters(),
        deltaTime,
        _particlesPerBatch);
}

auto SphSimulationENS::getParticleCountPerBatch() const -> uint32_t
{
    return _particlesPerBatch;
}

auto SphSimulationENS::getSharedMemorySize() const -> size_t
{
    return _sharedMemorySize;
}

}