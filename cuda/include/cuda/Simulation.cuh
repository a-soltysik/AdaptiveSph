#pragma once

#include <glm/ext/vector_float3.hpp>
#include <memory>
#include <vector>

#include "Api.cuh"
#include "ImportedMemory.cuh"

namespace sph::cuda
{

struct SPH_CUDA_API ParticleData
{
    alignas(16) glm::vec3 position;
    alignas(16) glm::vec3 velocity;
    alignas(16) glm::vec3 force;
    float mass;
};

class SPH_CUDA_API Simulation
{
public:
    struct SimulationData
    {
        struct Domain
        {
            glm::vec3 min;
            glm::vec3 max;
        };

        Domain domain;
        float density;
        float restitution;
    };

    struct FrameData
    {
        float deltaTime;
    };

    virtual ~Simulation() = default;

    virtual void update(FrameData data) = 0;
};

SPH_CUDA_API auto createSimulation(const Simulation::SimulationData& data,
                                   const std::vector<glm::vec3>& positions,
                                   const ImportedMemory& memory) -> std::unique_ptr<Simulation>;

}
