#include "Kernel.cuh"
#include "glm/exponential.hpp"
#include "glm/ext/scalar_constants.hpp"

namespace sph::cuda::device
{

__device__ __constant__ float PI_CONST = 3.14159265359f;
__device__ __constant__ float WENDLAND_NORM = 21.0f / (16.0f * 3.14159265359f);

__device__ auto wendlandLaplacianKernel(float distance, float smoothingRadius) -> float
{
    if (distance < smoothingRadius)
    {
        const float q = distance / smoothingRadius;
        const float h = smoothingRadius;

        const float factor = 105.0F / (16.0F * glm::pi<float>() * glm::pow(h, 5.0F));
        const float oneMq = 1.0F - q;

        return factor * oneMq * oneMq * (1.0F - 5.0F * q);
    }
    return 0.F;
}

__device__ auto wendlandKernel(float distance, float smoothingRadius) -> float
{
    const float q = distance / smoothingRadius;
    if (q > 2.0F)
    {
        return 0.0F;
    }

    const float volumeFactor = 1.0F / (smoothingRadius * smoothingRadius * smoothingRadius);
    const float tmp = 1.0F - (0.5F * q);
    return WENDLAND_NORM * volumeFactor * tmp * tmp * tmp * tmp * (2.0F * q + 1.0F);
}

__device__ auto wendlandDerivativeKernel(float distance, float smoothingRadius) -> float
{
    const float q = distance / smoothingRadius;
    if (q > 2.0F)
    {
        return 0.0F;
    }

    const float volumeFactor = 1.0F / (smoothingRadius * smoothingRadius * smoothingRadius);
    const float tmp = 1.0F - (0.5F * q);

    return WENDLAND_NORM * volumeFactor * (-5.0F * q * tmp * tmp * tmp) / smoothingRadius;
}

__device__ auto densityKernel(float distance, float smoothingRadius) -> float
{
    return wendlandKernel(distance, smoothingRadius);
}

__device__ auto densityDerivativeKernel(float distance, float smoothingRadius) -> float
{
    return wendlandDerivativeKernel(distance, smoothingRadius);
}

__device__ auto densityLaplacianKernel(float distance, float smoothingRadius) -> float
{
    return wendlandLaplacianKernel(distance, smoothingRadius);
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
    if (distance < smoothingRadius)
    {
        return 45.F / (glm::pi<float>() * glm::pow(smoothingRadius, 6.F)) * (smoothingRadius - distance);
    }
    return 0.F;
}

}
