#pragma once

#include <../../cuda/include/cuda/simulation/Simulation.cuh>
#include <cuda/refinement/RefinementParameters.cuh>
#include <glm/detail/qualifier.hpp>
#include <nlohmann/adl_serializer.hpp>
#include <nlohmann/detail/macro_scope.hpp>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <optional>

#include "utils/Configuration.hpp"

namespace nlohmann
{
template <typename T, glm::qualifier Q>
struct adl_serializer<glm::vec<3, T, Q>>
{
    static void to_json(json& j, const glm::vec<3, T, Q>& vec)
    {
        j["x"] = vec.x;
        j["y"] = vec.y;
        j["z"] = vec.z;
    }

    static void from_json(const json& j, glm::vec<3, T, Q>& vec)
    {
        vec.x = j["x"].get<T>();
        vec.y = j["y"].get<T>();
        vec.z = j["z"].get<T>();
    }
};

template <typename T>
struct adl_serializer<std::optional<T>>
{
    static void to_json(json& j, const std::optional<T>& opt)
    {
        if (opt == std::nullopt)
        {
            j = nullptr;
        }
        else
        {
            j = *opt;
        }
    }

    static void from_json(const json& j, std::optional<T>& opt)
    {
        if (j.is_null())
        {
            opt = std::nullopt;
        }
        else
        {
            opt = j.get<T>();
        }
    }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(sph::cuda::refinement::VelocityParameters::Split, minimalSpeedThreshold)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(sph::cuda::refinement::VelocityParameters::Merge, maximalSpeedThreshold)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(sph::cuda::refinement::VelocityParameters, split, merge)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(sph::cuda::refinement::VorticityParameters::Split, minimalVorticityThreshold)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(sph::cuda::refinement::VorticityParameters::Merge, maximalVorticityThreshold)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(sph::cuda::refinement::VorticityParameters, split, merge)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(sph::cuda::refinement::CurvatureParameters::Split, minimalCurvatureThreshold)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(sph::cuda::refinement::CurvatureParameters::Merge, maximalCurvatureThreshold)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(sph::cuda::refinement::CurvatureParameters, split, merge)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(sph::cuda::refinement::InterfaceParameters::Split, distanceRatioThreshold)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(sph::cuda::refinement::InterfaceParameters::Merge, distanceRatioThreshold)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(sph::cuda::refinement::InterfaceParameters, split, merge)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    sph::cuda::refinement::SplittingParameters, epsilon, alpha, centerMassRatio, vertexMassRatio)
NLOHMANN_JSON_SERIALIZE_ENUM(sph::cuda::refinement::RefinementParameters::Criterion,
                             {
                                 {sph::cuda::refinement::RefinementParameters::Criterion::Velocity,  "velocity" },
                                 {sph::cuda::refinement::RefinementParameters::Criterion::Interface, "interface"},
                                 {sph::cuda::refinement::RefinementParameters::Criterion::Vorticity, "vorticity"}
})
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(sph::cuda::refinement::RefinementParameters,
                                   enabled,
                                   minMassRatio,
                                   maxMassRatio,
                                   maxParticleCount,
                                   maxBatchRatio,
                                   initialCooldown,
                                   cooldown,
                                   criterionType,
                                   splitting,
                                   velocity,
                                   vorticity,
                                   curvature,
                                   interfaceParameters)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(sph::cuda::Simulation::Parameters::Domain, min, max, friction)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(sph::cuda::Simulation::Parameters,
                                   domain,
                                   gravity,
                                   speedOfSound,
                                   restDensity,
                                   restitution,
                                   viscosityConstant,
                                   surfaceTensionConstant,
                                   maxVelocity,
                                   baseSmoothingRadius,
                                   baseParticleRadius,
                                   baseParticleMass,
                                   threadsPerBlock)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(sph::utils::InitialParameters, particleCount)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(sph::utils::RenderingParameters, renderBoundaryParticles)

template <>
struct adl_serializer<sph::utils::Configuration>
{
    static void to_json(json& j, const sph::utils::Configuration& c)
    {
        j = json {
            {"initialParameters",    c.initialParameters   },
            {"simulationParameters", c.simulationParameters},
            {"refinementParameters", c.refinementParameters},
            {"renderingParameters",  c.renderingParameters }
        };
    }

    static void from_json(const json& j, sph::utils::Configuration& c)
    {
        c.initialParameters = j.value("initialParameters", std::optional<sph::utils::InitialParameters> {});
        c.simulationParameters = j.value("simulationParameters", std::optional<sph::cuda::Simulation::Parameters> {});
        c.refinementParameters =
            j.value("refinementParameters", std::optional<sph::cuda::refinement::RefinementParameters> {});
        c.renderingParameters = j.value("renderingParameters", std::optional<sph::utils::RenderingParameters> {});
    }
};

}
