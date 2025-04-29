#include "../SphSimulation.cuh"
#include "Utils.cuh"
#include "glm/vec3.hpp"

namespace sph::cuda
{

template <typename Func>
__device__ void forEachNeighbour(glm::vec4 position,
                                 const SphSimulation::Parameters& simulationData,
                                 const SphSimulation::Grid& grid,
                                 Func&& func)
{
    const auto originCell = calculateCellIndex(position, simulationData, grid);
    for (const auto offset : offsets)
    {
        const auto range = getStartEndIndices(originCell + glm::uvec3 {offset.x, offset.y, offset.z}, grid);

        if (range.first == -1 || range.first > range.second)
        {
            continue;
        }

        for (auto i = range.first; i <= range.second; i++)
        {
            const auto neighborIdx = grid.particleArrayIndices.data[i];

            func(neighborIdx);
        }
    }
}

}
