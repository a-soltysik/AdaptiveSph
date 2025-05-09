#pragma once

namespace sph::cuda::refinement
{

struct VelocityParameters
{
    struct Split
    {
        float minimalSpeedThreshold;
    };

    struct Merge
    {
        float maximalSpeedThreshold;
    };

    Split split;
    Merge merge;
};

}
