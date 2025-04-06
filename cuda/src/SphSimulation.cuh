#pragma once

#include <vector_types.h>

#include <cstddef>
#include <cstdint>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_uint3.hpp>
#include <glm/glm.hpp>
#include <vector>

#include "ImportedParticleMemory.cuh"
#include "Span.cuh"
#include "cuda/ImportedMemory.cuh"
#include "cuda/Simulation.cuh"

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
                  const std::vector<glm::vec3>& positions,
                  const ImportedMemory& memory);

    ~SphSimulation() override;

    void update(const Parameters& parameters, float deltaTime) override;

private:
    static auto createGrid(const Parameters& data, size_t particleCount) -> Grid;
    [[nodiscard]] auto getBlocksPerGridForParticles() const -> dim3;
    [[nodiscard]] auto getBlocksPerGridForGrid() const -> dim3;
    [[nodiscard]] auto getParticles() const -> Span<ParticleData>;

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

    const ImportedParticleMemory& _particleBuffer;
    Parameters _simulationData;
    State _state;
    size_t _particleCount = 0;
};
}
