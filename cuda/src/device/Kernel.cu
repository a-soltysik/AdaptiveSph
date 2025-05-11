#include "Kernel.cuh"
#include "glm/exponential.hpp"
#include "glm/ext/scalar_constants.hpp"

namespace sph::cuda::device
{
__device__ auto densityKernel(float distance, float smoothingRadius) -> float
{
    const float q = distance / smoothingRadius;
    if (q > 2.0f)
    {
        return 0.0f;
    }

    const float normalization = 21.0f / (16.0f * glm::pi<float>());
    const float volumeFactor = 1.0f / (smoothingRadius * smoothingRadius * smoothingRadius);
    const float tmp = 1.0f - 0.5f * q;
    return normalization * volumeFactor * tmp * tmp * tmp * tmp * (2.0f * q + 1.0f);
}

__device__ auto densityDerivativeKernel(float distance, float smoothingRadius) -> float
{
    const float q = distance / smoothingRadius;
    if (q > 2.0f || distance < 1e-6f)
    {
        return 0.0f;
    }

    const float normalization = 21.0f / (16.0f * glm::pi<float>());
    const float volumeFactor = 1.0f / (smoothingRadius * smoothingRadius * smoothingRadius);
    const float tmp = 1.0f - 0.5f * q;

    return normalization * volumeFactor * (-5.0f * q * tmp * tmp * tmp) / smoothingRadius;
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

__device__ auto wendlandLaplacianKernel(float distance, float smoothingRadius) -> float
{
    if (distance < smoothingRadius)
    {
        const float q = distance / smoothingRadius;
        const float h = smoothingRadius;

        const float factor = 105.0f / (16.0f * glm::pi<float>() * glm::pow(h, 5.0f));
        const float oneMq = 1.0f - q;

        return factor * oneMq * oneMq * (1.0f - 5.0f * q);
    }
    return 0.F;
}
}
