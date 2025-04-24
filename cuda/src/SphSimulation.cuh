#pragma once

#include <vector_types.h>

#include <cstddef>
#include <cstdint>
#include <cuda/ImportedMemory.cuh>
#include <cuda/Simulation.cuh>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_uint3.hpp>
#include <glm/glm.hpp>
#include <vector>

#include "ImportedParticleMemory.cuh"
#include "Span.cuh"

namespace sph::cuda
{

class SphSimulation : public Simulation
{
public:
    struct Grid
    {
        glm::uvec3 gridSize;
        glm::vec3 cellSize;
        Span<int32_t> cellStartIndices;
        Span<int32_t> cellEndIndices;
        Span<int32_t> particleGridIndices;
        Span<int32_t> particleArrayIndices;
    };

    struct State
    {
        Grid grid;
    };

    SphSimulation(const Parameters& initialParameters,
                  const std::vector<glm::vec4>& positions,
                  const ParticlesDataBuffer& memory);

    ~SphSimulation() override;

    void update(const Parameters& parameters, float deltaTime) override;

private:
    struct ParticlesInternalDataBuffer
    {
        const ImportedParticleMemory& positions;
        const ImportedParticleMemory& predictedPositions;
        const ImportedParticleMemory& velocities;
        const ImportedParticleMemory& forces;
        const ImportedParticleMemory& densities;
        const ImportedParticleMemory& nearDensities;
        const ImportedParticleMemory& pressures;
        const ImportedParticleMemory& radiuses;
        const ImportedParticleMemory& masses;
    };

    static auto createGrid(const Parameters& data, size_t particleCount) -> Grid;
    static auto toInternalBuffer(const ParticlesDataBuffer& memory) -> ParticlesInternalDataBuffer;
    static auto getParticleMass(float domainVolume, float restDensity, uint32_t particlesCount) -> float;

    [[nodiscard]] auto getBlocksPerGridForParticles() const -> dim3;
    [[nodiscard]] auto getBlocksPerGridForGrid() const -> dim3;
    [[nodiscard]] auto getParticles() const -> ParticlesData;

    void computeExternalForces(float deltaTime) const;
    void resetGrid() const;
    void assignParticlesToCells() const;
    void sortParticles() const;
    void calculateCellStartAndEndIndices() const;
    void computeDensities() const;
    void computePressureForce(float deltaTime) const;
    void computeViscosityForce(float deltaTime) const;
    void integrateMotion(float deltaTime) const;
    void handleCollisions() const;

    ParticlesInternalDataBuffer _particleBuffer;
    Parameters _simulationData;
    State _state;
    uint32_t _particleCount = 0;
};
}
