#include "Kernel.cuh"
#include "glm/exponential.hpp"
#include "glm/ext/scalar_constants.hpp"

namespace sph::cuda::device
{

__device__ auto densityKernel(float distance, float smoothingRadius) -> float
{
    if (distance < smoothingRadius)
    {
        const auto scale = 15.F / (2.F * glm::pi<float>() * glm::pow(smoothingRadius, 5.F));
        const auto v = smoothingRadius - distance;
        return v * v * scale;
    }
    return 0.F;
}

__device__ auto densityDerivativeKernel(float distance, float smoothingRadius) -> float
{
    if (distance < smoothingRadius)
    {
        const auto scale = 15.F / (glm::pi<float>() * glm::pow(smoothingRadius, 5.F));
        const auto v = smoothingRadius - distance;
        return -v * scale;
    }
    return 0.F;
}

__device__ auto nearDensityKernel(float distance, float smoothingRadius) -> float
{
    if (distance < smoothingRadius)
    {
        const auto scale = 15.F / (glm::pi<float>() * glm::pow(smoothingRadius, 6.F));
        const auto v = smoothingRadius - distance;
        return v * v * v * scale;
    }
    return 0.F;
}

__device__ auto nearDensityDerivativeKernel(float distance, float smoothingRadius) -> float
{
    if (distance < smoothingRadius)
    {
        const auto scale = 45.F / (glm::pi<float>() * glm::pow(smoothingRadius, 6.F));
        const auto v = smoothingRadius - distance;
        return -v * v * scale;
    }
    return 0.F;
}

__device__ auto smoothingKernelPoly6(float distance, float smoothingRadius) -> float
{
    if (distance < smoothingRadius)
    {
        const auto scale = 315.F / (64.F * glm::pi<float>() * glm::pow(smoothingRadius, 9.F));
        const auto v = smoothingRadius * smoothingRadius - distance * distance;
        return v * v * v * scale;
    }
    return 0.F;
}

}
