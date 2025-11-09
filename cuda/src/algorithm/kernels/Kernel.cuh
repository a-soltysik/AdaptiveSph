#pragma once

namespace sph::cuda::device
{

namespace detail
{
namespace constants
{
constexpr auto pi = 3.141592653589793F;
constexpr auto wendlandCoefficient = 21.F / (16.F * pi);
constexpr auto wendlandLaplacianCoefficient = 105.F / (16.F * pi);
constexpr auto wendlandDerivativeCoefficient = -5.F * wendlandCoefficient;
constexpr auto nearDensityCoefficient = 15.F / pi;
constexpr auto nearDensityDerivativeCoefficient = 45.F / pi;
constexpr auto viscosityCoefficient = 15.F / (2.F * pi);
constexpr auto viscosityLaplacianCoefficient = 45.F / pi;
}

__device__ __forceinline__ auto pow2(float x) -> float
{
    return x * x;
}

__device__ __forceinline__ auto pow3(float x) -> float
{
    return x * x * x;
}

__device__ __forceinline__ auto pow5(float x) -> float
{
    const auto x2 = pow2(x);
    return x2 * x2 * x;
}

__device__ __forceinline__ auto pow6(float x) -> float
{
    const auto x3 = pow3(x);
    return pow2(x3);
}
}

__forceinline__ __device__ auto wendlandLaplacianKernel(float distance, float smoothingRadius) -> float
{
    if (distance < smoothingRadius)
    {
        const auto q = distance / smoothingRadius;
        const auto h5 = detail::pow5(smoothingRadius);

        const auto oneMq = 1.0F - q;
        const auto oneMq2 = oneMq * oneMq;

        return (detail::constants::wendlandLaplacianCoefficient / h5) * oneMq2 * (1.0F - 5.0F * q);
    }
    return 0.0F;
}

__forceinline__ __device__ auto wendlandKernel(float distance, float smoothingRadius) -> float
{
    const float q = distance / smoothingRadius;
    if (q > 2.0F)
    {
        return 0.0F;
    }

    const auto h3 = detail::pow3(smoothingRadius);
    const auto tmp = 1.0F - 0.5F * q;
    const auto tmp4 = detail::pow2(tmp) * detail::pow2(tmp);

    return (detail::constants::wendlandCoefficient / h3) * tmp4 * (2.0F * q + 1.0F);
}

__forceinline__ __device__ auto wendlandDerivativeKernel(float distance, float smoothingRadius) -> float
{
    const float q = distance / smoothingRadius;
    if (q > 2.0F || distance < 1e-6F)
    {
        return 0.0F;
    }

    const float h3 = detail::pow3(smoothingRadius);
    const float h4 = h3 * smoothingRadius;

    const float tmp = 1.0F - 0.5F * q;
    const float tmp3 = tmp * tmp * tmp;

    return (detail::constants::wendlandDerivativeCoefficient / h4) * q * tmp3;
}

__forceinline__ __device__ auto densityKernel(float distance, float smoothingRadius) -> float
{
    return wendlandKernel(distance, smoothingRadius);
}

__forceinline__ __device__ auto densityDerivativeKernel(float distance, float smoothingRadius) -> float
{
    return wendlandDerivativeKernel(distance, smoothingRadius);
}

__forceinline__ __device__ auto densityLaplacianKernel(float distance, float smoothingRadius) -> float
{
    return wendlandLaplacianKernel(distance, smoothingRadius);
}

__forceinline__ __device__ auto nearDensityKernel(float distance, float smoothingRadius) -> float
{
    if (distance < smoothingRadius)
    {
        const float h6 = detail::pow6(smoothingRadius);
        const float v = smoothingRadius - distance;
        const float v3 = v * v * v;

        return (detail::constants::nearDensityCoefficient / h6) * v3;
    }
    return 0.F;
}

__forceinline__ __device__ auto nearDensityDerivativeKernel(float distance, float smoothingRadius) -> float
{
    if (distance < smoothingRadius)
    {
        const float h6 = detail::pow6(smoothingRadius);
        const float v = smoothingRadius - distance;
        const float v2 = v * v;

        return -(detail::constants::nearDensityDerivativeCoefficient / h6) * v2;
    }
    return 0.F;
}

__forceinline__ __device__ auto viscosityKernel(float distance, float smoothingRadius) -> float
{
    if (distance < smoothingRadius)
    {
        const float h = smoothingRadius;
        const float h2 = h * h;
        const float h3 = h2 * h;

        const float d = distance;
        const float d2 = d * d;
        const float d3 = d2 * d;

        const float cubic = -d3 / (2.0F * h3);
        const float quadratic = d2 / h2;
        const float linear = h / (2.0F * d) - 1.0F;

        return (detail::constants::viscosityCoefficient / h3) * (cubic + quadratic + linear);
    }
    return 0.F;
}

__forceinline__ __device__ auto viscosityLaplacianKernel(float distance, float smoothingRadius) -> float
{
    const float h6 = detail::pow6(smoothingRadius);
    return (detail::constants::viscosityLaplacianCoefficient / h6) * (smoothingRadius - distance);
}

}
