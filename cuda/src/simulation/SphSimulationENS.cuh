#pragma once
#include "SphSimulation.cuh"
#include "utils/NeighborSharingUtils.cuh"

namespace sph::cuda
{

class SphSimulationENS : public SphSimulation
{
public:
    SphSimulationENS(const Parameters& initialParameters,
                     const std::vector<glm::vec4>& positions,
                     const ParticlesDataBuffer& memory,
                     uint32_t maxParticleCapacity);

    ~SphSimulationENS() override;

    void update(float deltaTime) override;

    auto getParticleCountPerBatch() const -> uint32_t;
    auto getSharedMemorySize() const -> size_t;

protected:
    void computeParticleRectangles() const;
    void computeDensitiesENS() const;
    void computePressureForceENS(float deltaTime) const;
    void computeViscosityForceENS(float deltaTime) const;

private:
    void calculateOptimalConfiguration();
    void allocateEnsMemory(uint32_t maxParticleCapacity);
    void freeEnsMemory();

    Rectangle3D* _particleRectangles = nullptr;

    uint32_t _particlesPerBatch;
    size_t _sharedMemorySize;
};
}
