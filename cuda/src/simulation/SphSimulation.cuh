#pragma once

#include <thrust/device_vector.h>
#include <vector_types.h>

#include <cstddef>
#include <cstdint>
#include <cuda/Simulation.cuh>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/ext/vector_uint3.hpp>
#include <glm/glm.hpp>
#include <span>
#include <vector>

#include "memory/ImportedParticleMemory.cuh"

namespace sph::cuda
{

class SphSimulation : public Simulation
{
public:
    struct Grid
    {
        glm::uvec3 gridSize;
        glm::vec3 cellSize;
        std::span<int32_t> cellStartIndices;
        std::span<int32_t> cellEndIndices;
        std::span<int32_t> particleGridIndices;
        std::span<int32_t> particleArrayIndices;
    };

    SphSimulation(const Parameters& initialParameters,
                  const std::vector<glm::vec4>& positions,
                  const ParticlesDataBuffer& memory,
                  uint32_t maxParticleCapacity);

    SphSimulation(const SphSimulation&) = delete;
    SphSimulation(SphSimulation&&) = delete;

    auto operator=(const SphSimulation&) -> SphSimulation& = delete;
    auto operator=(SphSimulation&&) -> SphSimulation& = delete;
    ~SphSimulation() override;

    void update(float deltaTime) override;

    [[nodiscard]] auto getParticlesCount() const -> uint32_t override
    {
        return _particleCount;
    }

    [[nodiscard]] auto calculateAverageNeighborCount() const -> float override;
    [[nodiscard]] auto getDensityInfo(float threshold) const -> DensityInfo override;

protected:
    struct ParticlesInternalDataBuffer
    {
        const ImportedParticleMemory& positions;
        const ImportedParticleMemory& predictedPositions;
        const ImportedParticleMemory& velocities;
        const ImportedParticleMemory& densities;
        const ImportedParticleMemory& nearDensities;
        const ImportedParticleMemory& pressures;
        const ImportedParticleMemory& radiuses;
        const ImportedParticleMemory& smoothingRadiuses;
        const ImportedParticleMemory& masses;
    };

    static auto createGrid(const Parameters& data, size_t particleCapacity) -> Grid;
    static auto toInternalBuffer(const ParticlesDataBuffer& memory) -> ParticlesInternalDataBuffer;

    [[nodiscard]] auto getParticles() const -> ParticlesData;

    [[nodiscard]] auto getGrid() const -> const Grid&
    {
        return _grid;
    }

    [[nodiscard]] auto getParameters() const -> const Parameters&
    {
        return _simulationData;
    }

    [[nodiscard]] auto getThreadsPerBlock() const -> uint32_t
    {
        return _simulationData.threadsPerBlock;
    }

    void updateParameters(const Parameters& parameters)
    {
        _simulationData = parameters;
    }

    void setParticleCount(uint32_t count)
    {
        _particleCount = count;
    }

    [[nodiscard]] auto getParticlesCapacity() const -> uint32_t
    {
        return _particleCapacity;
    }

    [[nodiscard]] auto getBlocksPerGridForParticles() const -> dim3;
    [[nodiscard]] auto getBlocksPerGridForGrid() const -> dim3;

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

private:
    ParticlesInternalDataBuffer _particleBuffer;
    Parameters _simulationData;
    Grid _grid;
    uint32_t _particleCount = 0;
    uint32_t _particleCapacity = 0;
};
}
