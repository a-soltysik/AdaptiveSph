#pragma once

namespace sph::cuda::kernel
{

inline __device__ auto computeTaitPressure(float density, float restDensity, float speedOfSound) -> float
{
    static constexpr auto gamma = 5.F;
    const auto B = restDensity * speedOfSound * speedOfSound / gamma;
    const auto densityRatio = density / restDensity;
    return B * (powf(densityRatio, gamma) - 1.F);
}

}
