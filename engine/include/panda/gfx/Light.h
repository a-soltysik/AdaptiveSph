#pragma once

#include <glm/ext/vector_float3.hpp>
#include <string>
#include <vector>

namespace panda::gfx
{

inline constexpr auto maxLights = size_t {5};

struct Attenuation
{
    float constant;
    float linear;
    float exp;
};

struct BaseLight
{
    std::string name;
    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;
    float intensity;

    auto makeColorLight(
        glm::vec3 color, float newAmbient, float newDiffuse, float newSpecular, float newIntensity = 1.F) -> void
    {
        ambient = color * newAmbient;
        diffuse = color * newDiffuse;
        specular = color * newSpecular;
        intensity = newIntensity;
    }
};

struct DirectionalLight : BaseLight
{
    glm::vec3 direction;
};

struct PointLight : BaseLight
{
    glm::vec3 position;
    Attenuation attenuation;
};

struct SpotLight : PointLight
{
    glm::vec3 direction;
    float cutOff;
};

struct Lights
{
    std::vector<DirectionalLight> directionalLights;
    std::vector<PointLight> pointLights;
    std::vector<SpotLight> spotLights;
};

}
