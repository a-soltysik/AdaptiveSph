#include <algorithm>
#include <execution>
#include <glm/geometric.hpp>
#include <iterator>
#include <nanoflann.hpp>
#include <numeric>

#include "algorithm/kernels/Kernel.cuh"
#include "cuda/physics/StaticBoundaryDomain.cuh"

namespace sph::cuda::physics
{

namespace
{

struct PointCloud
{
    const std::vector<glm::vec4>& points;

    explicit PointCloud(const std::vector<glm::vec4>& pts)
        : points(pts)
    {
    }

    [[maybe_unused]] size_t kdtree_get_point_count() const
    {
        return points.size();
    }

    [[maybe_unused]] float kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        return points[idx][static_cast<int>(dim)];
    }

    template <class BBOX>
    [[maybe_unused]] bool kdtree_get_bbox(BBOX&) const
    {
        return false;
    }
};

using KDTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud>, PointCloud, 3>;

auto computeDeltaWithKDTree(const KDTree& kdtree, glm::vec4 particlePosition, float smoothingRadius) -> float
{
    const auto searchRadius = device::constant::wendlandRangeRatio * smoothingRadius;
    const auto searchRadiusSq = searchRadius * searchRadius;

    const float query_pt[3] = {particlePosition.x, particlePosition.y, particlePosition.z};

    std::vector<nanoflann::ResultItem<uint32_t, float>> matches;
    nanoflann::SearchParameters params;
    params.sorted = false;

    kdtree.radiusSearch(query_pt, searchRadiusSq, matches, params);

    float sum = 0.0F;
    for (const auto& match : matches)
    {
        const auto distance = std::sqrt(match.second);
        sum += device::wendlandKernel(distance, smoothingRadius);
    }

    return sum;
}
}

StaticBoundaryDomain::StaticBoundaryDomain(Simulation::Parameters::Domain bounds, std::vector<Particle> particles)
    : _bounds {bounds},
      _particles {std::move(particles)}
{
}

auto StaticBoundaryDomain::generate(const Simulation::Parameters::Domain& bounds,
                                    float particleSpacing,
                                    float fluidRestDensity,
                                    float fluidSmoothingRadius) -> StaticBoundaryDomain
{
    auto positions = generateWallParticles(bounds, particleSpacing);
    const auto smoothingRadius = fluidSmoothingRadius;
    auto particles = computePsi(positions, smoothingRadius, fluidRestDensity);

    return StaticBoundaryDomain {bounds, std::move(particles)};
}

auto StaticBoundaryDomain::generateWallParticles(const Simulation::Parameters::Domain& bounds, float spacing)
    -> std::vector<glm::vec4>
{
    std::vector<glm::vec4> positions;
    const auto particlesCount = bounds.getScale() / spacing;
    positions.reserve(static_cast<size_t>(particlesCount.x * particlesCount.y * particlesCount.z));

    const auto& min = bounds.min;
    const auto& max = bounds.max;

    for (float x = min.x; x < max.x; x += spacing)
    {
        for (float z = min.z; z < max.z; z += spacing)
        {
            positions.emplace_back(x, min.y, z, 0.0F);
        }
    }

    for (float x = min.x; x < max.x; x += spacing)
    {
        for (float z = min.z; z < max.z; z += spacing)
        {
            positions.emplace_back(x, max.y, z, 0.0F);
        }
    }

    for (float y = min.y + spacing; y < max.y; y += spacing)
    {
        for (float z = min.z; z < max.z; z += spacing)
        {
            positions.emplace_back(min.x, y, z, 0.0F);
        }
    }
    for (float y = min.y + spacing; y < max.y; y += spacing)
    {
        for (float z = min.z; z < max.z; z += spacing)
        {
            positions.emplace_back(max.x, y, z, 0.0F);
        }
    }

    for (float x = min.x + spacing; x < max.x; x += spacing)
    {
        for (float y = min.y + spacing; y < max.y; y += spacing)
        {
            positions.emplace_back(x, y, min.z, 0.0F);
        }
    }

    for (float x = min.x + spacing; x < max.x; x += spacing)
    {
        for (float y = min.y + spacing; y < max.y; y += spacing)
        {
            positions.emplace_back(x, y, max.z, 0.0F);
        }
    }

    return positions;
}

auto StaticBoundaryDomain::computePsi(const std::vector<glm::vec4>& positions, float smoothingRadius, float restDensity)
    -> std::vector<Particle>
{
    if (positions.empty())
    {
        return {};
    }

    PointCloud cloud(positions);
    KDTree kdtree(3, cloud, {10});

    std::vector<Particle> particles(positions.size());
    std::transform(std::execution::par_unseq,
                   positions.begin(),
                   positions.end(),
                   particles.begin(),
                   [&kdtree, smoothingRadius, restDensity](const auto& position) {
                       const auto delta = computeDeltaWithKDTree(kdtree, position, smoothingRadius);
                       const auto volume = 1.0F / delta;
                       const auto psi = restDensity * volume;

                       return Particle {.position = position, .psi = psi};
                   });

    return particles;
}

}
