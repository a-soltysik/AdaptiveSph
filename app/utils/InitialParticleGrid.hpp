#pragma once

#include <algorithm>
#include <numeric>
#include <utility>

#include "utils/Configuration.hpp"

namespace sph::utils
{
inline auto getInitialParticleGridRange(float min, float max, float fillRatio, float particleRadius, bool anchorAtMax)
    -> std::pair<float, float>
{
    const auto usableSpan = max - min - 2.F * initialParticleMargin;
    const auto requestedCenterSpan = fillRatio < 1.F ? fillRatio * (max - min) - 2.F * particleRadius : usableSpan;
    const auto centerSpan = std::clamp(requestedCenterSpan, 0.F, usableSpan);
    if (anchorAtMax)
    {
        const auto upper = max - initialParticleMargin;
        return {upper - centerSpan, upper};
    }

    const auto lower = min + initialParticleMargin;
    return {lower, lower + centerSpan};
}

inline auto getInitialParticleGridSpacing(const std::pair<float, float>& range, uint32_t particleCount) -> float
{
    if (particleCount == 1)
    {
        return 0.F;
    }
    return (range.second - range.first) / static_cast<float>(particleCount - 1);
}

inline auto getInitialParticleGridCoordinate(const std::pair<float, float>& range,
                                             uint32_t particleCount,
                                             uint32_t particleIndex) -> float
{
    if (particleCount == 1)
    {
        return std::midpoint(range.first, range.second);
    }
    return range.first + getInitialParticleGridSpacing(range, particleCount) * static_cast<float>(particleIndex);
}
}
