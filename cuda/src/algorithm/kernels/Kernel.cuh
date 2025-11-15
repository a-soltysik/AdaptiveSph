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

}

namespace constant
{
constexpr auto wendlandRangeRatio = 2.F;
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

__forceinline__ __device__ auto wendlandDerivativeKernel(float distance, float smoothingRadius) -> float
{
    const float q = distance / smoothingRadius;
    if (q > 2.0F)
    {
        return 0.0F;
    }

    const float h3 = detail::pow3(smoothingRadius);
    const float h4 = h3 * smoothingRadius;

    const float tmp = 1.0F - 0.5F * q;
    const float tmp3 = tmp * tmp * tmp;

    return (detail::constants::wendlandDerivativeCoefficient / h4) * q * tmp3;
}

}
