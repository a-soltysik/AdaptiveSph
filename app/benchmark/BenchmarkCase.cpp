/*#include "BenchmarkCase.hpp"

#include "DamBreakBenchmark.hpp"
#include "LidDrivenCavityBenchmark.hpp"
#include "PoiseuilleBenchmark.hpp"
#include "TaylorGreenBenchmark.hpp"
#include "panda/Logger.h"

namespace sph
{
std::unique_ptr<BenchmarkCase> BenchmarkFramework::createTestCase(const std::string& testName)
{
    if (testName == "poiseuille")
    {
        return std::make_unique<PoiseuilleBenchmark>();
    }
    if (testName == "taylorGreen")
    {
        return std::make_unique<TaylorGreenBenchmark>();
    }
    if (testName == "damBreak")
    {
        return std::make_unique<DamBreakBenchmark>();
    }
    if (testName == "lidDrivenCavity")
    {
        return std::make_unique<LidDrivenCavityBenchmark>();
    }

    panda::log::Error("Unknown test case: {}", testName);
    return nullptr;
}
}
*/
