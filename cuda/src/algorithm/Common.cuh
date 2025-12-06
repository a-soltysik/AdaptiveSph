#pragma once

namespace sph::cuda::kernel
{

namespace constants
{

constexpr auto minimalDistance = 0.00001F;

}

inline __device__ auto computeTaitPressure(float density, float restDensity, float speedOfSound) -> float
{
    static constexpr auto gamma = 5.F;
    const auto B = restDensity * speedOfSound * speedOfSound / gamma;
    const auto densityRatio = density / restDensity;
    return B * (powf(densityRatio, gamma) - 1.F);
}

}
