#pragma once

#include <thrust/device_vector.h>
#include <vector_types.h>

#include <cstdint>
#include <cuda/Simulation.cuh>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/glm.hpp>
#include <memory>
#include <span>
#include <vector>

#include "algorithm/NeighborGrid.cuh"
#include "memory/ImportedParticleMemory.cuh"

namespace sph::cuda::physics
{
class StaticBoundaryDomain;
}

namespace sph::cuda
{

class SphSimulation : public Simulation
{
public:
    SphSimulation(const Parameters& initialParameters,
                  const std::vector<glm::vec4>& positions,
                  const FluidParticlesDataImportedBuffer& fluidParticleMemory,
                  const BoundaryParticlesDataImportedBuffer& boundaryParticleMemory,
                  const physics::StaticBoundaryDomain& boundaryDomain,
                  uint32_t maxParticleCapacity);

    void update(float deltaTime) override;
    void updateDomain(const Parameters::Domain& domain, const physics::StaticBoundaryDomain& boundaryDomain) override;

    [[nodiscard]] auto getFluidParticlesCount() const -> uint32_t final
    {
        return _particleCount;
    }

    [[nodiscard]] auto getBoundaryParticlesCount() const -> uint32_t final
    {
        return _boundaryParticleCount;
    }

    [[nodiscard]] auto calculateAverageNeighborCount() -> float override;
    [[nodiscard]] auto getDensityInfo(float threshold) -> DensityInfo override;

protected:
    struct FluidParticlesDataBuffer
    {
        struct Imported
        {
            const ImportedParticleMemory& positions;
            const ImportedParticleMemory& velocities;
            const ImportedParticleMemory& densities;
            const ImportedParticleMemory& radii;
        };

        struct Internal
        {
            thrust::device_vector<glm::vec4> accelerations;
            thrust::device_vector<float> smoothingRadii;
            thrust::device_vector<float> masses;
        };

        Imported imported;
        Internal internal;
    };

    struct BoundaryParticlesDataBuffer
    {
        struct Imported
        {
            const ImportedParticleMemory& positions;
            const ImportedParticleMemory& radii;
            const ImportedParticleMemory& colors;
        };

        struct Internal
        {
            thrust::device_vector<float> psiValues;
            thrust::device_vector<float> viscosityCoefficients;
        };

        Imported imported;
        Internal internal;
    };

    static auto toInternalBuffer(const FluidParticlesDataImportedBuffer& memory) -> FluidParticlesDataBuffer::Imported;
    static auto toInternalBuffer(const BoundaryParticlesDataImportedBuffer& memory)
        -> BoundaryParticlesDataBuffer::Imported;

    [[nodiscard]] auto getFluidParticles() -> FluidParticlesData;
    [[nodiscard]] auto getBoundaryParticles() -> BoundaryParticlesData;

    [[nodiscard]] auto getGrid() const -> const NeighborGrid&
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

    [[nodiscard]] auto getBlocksPerGridForFluidParticles() const -> dim3;
    [[nodiscard]] auto getBlocksPerGridForBoundaryParticles() const -> dim3;

    void computeExternalAccelerations();
    void computeDensities();
    void computePressureAccelerations();
    void computeViscosityAccelerations();
    void computeSurfaceTensionAccelerations();
    void halfKickVelocities(float halfDt);
    void updatePositions(float deltaTime);
    void computeBoundaryDensityContribution();
    void computeBoundaryForces();

private:
    void initializeBoundaryParticles(const physics::StaticBoundaryDomain& boundaryDomain);

    FluidParticlesDataBuffer _fluidParticlesData;
    BoundaryParticlesDataBuffer _boundaryParticlesData;
    Parameters _simulationData;
    uint32_t _boundaryParticleCount;
    uint32_t _particleCount;
    uint32_t _particleCapacity;
    NeighborGrid _grid;
};
}
