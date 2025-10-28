#pragma once

// clang-format off
#include "panda/utils/Assert.h"
// clang-format on

#include <glm/ext/vector_float3.hpp>

#include "Alignment.h"
#include "panda/gfx/Light.h"

namespace panda::gfx
{

struct UboAttenuation
{
    float constant;
    float linear;
    float exp;

    PD_MAKE_ALIGNED(constant, linear, exp)
};

static_assert(alignment(UboAttenuation {}) == 16);

struct UboBaseLight
{
    alignas(16) glm::vec3 ambient;
    alignas(16) glm::vec3 diffuse;
    alignas(16) glm::vec3 specular;
    float intensity;

    PD_MAKE_ALIGNED(ambient, diffuse, specular, intensity)
};

static_assert(alignment(UboBaseLight {}) == 16);

struct UboDirectionalLight
{
    UboBaseLight base;
    alignas(16) glm::vec3 direction;

    PD_MAKE_ALIGNED(base, direction)
};

struct UboPointLight
{
    alignas(16) UboBaseLight base;
    alignas(16) UboAttenuation attenuation;
    alignas(16) glm::vec3 position;

    PD_MAKE_ALIGNED(base, attenuation, position)
};

struct UboSpotLight
{
    UboPointLight base;
    alignas(16) glm::vec3 direction;
    float cutOff;

    PD_MAKE_ALIGNED(base, direction, cutOff)
};

constexpr auto fromDirectionalLight(const DirectionalLight& light) noexcept -> UboDirectionalLight
{
    return {
        .base = {.ambient = light.ambient,
                 .diffuse = light.diffuse,
                 .specular = light.specular,
                 .intensity = light.intensity},
        .direction = light.direction
    };
}

constexpr auto fromPointLight(const PointLight& light) noexcept -> UboPointLight
{
    return {
        .base = {.ambient = light.ambient,
                 .diffuse = light.diffuse,
                 .specular = light.specular,
                 .intensity = light.intensity},
        .attenuation = {.constant = light.attenuation.constant,
                 .linear = light.attenuation.linear,
                 .exp = light.attenuation.exp},
        .position = light.position
    };
}

constexpr auto fromSpotLight(const SpotLight& light) noexcept -> UboSpotLight
{
    return {
        .base = {.base = {.ambient = light.ambient,
                          .diffuse = light.diffuse,
                          .specular = light.specular,
                          .intensity = light.intensity},
                 .attenuation = {.constant = light.attenuation.constant,
                                 .linear = light.attenuation.linear,
                                 .exp = light.attenuation.exp},
                 .position = light.position},
        .direction = light.direction,
        .cutOff = light.cutOff
    };
}

}
