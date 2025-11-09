#pragma once

#include <cstdint>
#include <glm/ext/vector_int3.hpp>

#include "Utils.cuh"

namespace sph::cuda
{
struct Rectangle3D
{
    glm::ivec3 min;
    glm::ivec3 max;

    __device__ __host__ auto isEmpty() const -> bool
    {
        return min.x > max.x || min.y > max.y || min.z > max.z;
    }

    __device__ __host__ auto getCellCount() const -> uint32_t
    {
        if (isEmpty())
        {
            return 0;
        }
        return (max.x - min.x + 1) * (max.y - min.y + 1) * (max.z - min.z + 1);
    }

    __device__ __host__ auto contains(glm::ivec3 point) const -> bool
    {
        return point.x >= min.x && point.x <= max.x && point.y >= min.y && point.y <= max.y && point.z >= min.z &&
               point.z <= max.z;
    }
};

struct CompactParticleData
{
    glm::vec4 predictedPosition;
    float smoothingRadius;
    float mass;
    int32_t originalIndex;
};

struct DeviceCapabilities
{
    int maxSharedMemoryPerBlock;
    int maxSharedMemoryPerMultiprocessor;
    int maxThreadsPerBlock;

    static auto query(int deviceId = 0) -> DeviceCapabilities
    {
        DeviceCapabilities caps {};
        cudaDeviceGetAttribute(&caps.maxSharedMemoryPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, deviceId);
        cudaDeviceGetAttribute(&caps.maxSharedMemoryPerMultiprocessor,
                               cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                               deviceId);
        cudaDeviceGetAttribute(&caps.maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, deviceId);

        return caps;
    }
};

inline auto calculateParticlesPerBatch(int32_t threadsPerBlock, uint32_t availableSharedMemory) -> uint32_t
{
    const auto warpsPerBlock = (threadsPerBlock + constants::warpSize - 1) / constants::warpSize;
    const auto sharedMemoryPerWarp = availableSharedMemory / warpsPerBlock;
    return sharedMemoryPerWarp / sizeof(CompactParticleData);
}

}