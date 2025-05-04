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

__device__ auto viscosityKernel(float distance, float smoothingRadius) -> float
{
    if (distance < smoothingRadius)
    {
        const auto scale = 15.F / (2.F * glm::pi<float>() * glm::pow(smoothingRadius, 3.F));
        const auto cubic =
            -distance * distance * distance / (2.F * smoothingRadius * smoothingRadius * smoothingRadius);
        const auto quadratic = distance * distance / (smoothingRadius * smoothingRadius);
        const auto linear = (smoothingRadius / (2.F * distance)) - 1;
        return scale * (cubic + quadratic + linear);
    }
    return 0.F;
}

__device__ auto viscosityLaplacianKernel(float distance, float smoothingRadius) -> float
{
    return 45.F / (glm::pi<float>() * glm::pow(smoothingRadius, 6.F)) * (smoothingRadius - distance);
}

}
