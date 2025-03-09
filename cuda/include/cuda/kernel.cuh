#pragma once

#include <glm/ext/vector_float3.hpp>
#include <vector>

#ifdef _WIN32
#    ifdef EXPORTING_CUDA_LIB
#        define SPH_CUDA_API __declspec(dllexport)
#    else
#        define SPH_CUDA_API __declspec(dllimport)
#    endif
#else
#    define SPH_CUDA_API
#endif

namespace sph::cuda
{

struct SimulationData
{
    std::vector<glm::vec3> positions;
};

struct FrameData
{
    float deltaTime;
};

SPH_CUDA_API void initialize(const SimulationData& data);
SPH_CUDA_API void update(FrameData data);
SPH_CUDA_API void getUpdatedPositions(std::vector<glm::vec3*>& objects);

}
