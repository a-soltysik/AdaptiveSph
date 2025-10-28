#version 450

struct Attenuation
{
    float constant;
    float linear;
    float exp;
};

struct BaseLight
{
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float intensity;
};

struct DirectionalLight
{
    BaseLight base;
    vec3 direction;
};

struct PointLight
{
    BaseLight base;
    Attenuation attenuation;
    vec3 position;
};

struct SpotLight
{
    PointLight base;
    vec3 direction;
    float cutOff;
};

layout (location = 0) in vec2 uv;
layout (location = 1) in vec3 worldPosition;
layout (location = 2) in vec3 sphereCenter;
layout (location = 3) in float radius;
layout (location = 4) in vec3 velocity;
layout (location = 5) in float densityDeviation;

layout (location = 0) out vec4 outColor;

layout (set = 0, binding = 1) uniform GlobalUbo
{
    mat4 inverseView;

    PointLight pointLights[5];
    DirectionalLight directionalLights[5];
    SpotLight spotLights[5];

    vec3 globalAmbient;
    uint activePointLights;
    uint activeDirectionalLights;
    uint activeSpotLights;
} ubo;

vec3 calculateLight(BaseLight light, vec3 lightDirection, vec3 normal, vec3 intersectionPoint);
vec3 calculatePointLight(PointLight light, vec3 normal, vec3 intersectionPoint);
vec3 calculateDirectionalLight(DirectionalLight light, vec3 normal);
vec3 calculateSpotLight(SpotLight light, vec3 normal, vec3 intersectionPoint);
bool raySphereIntersection(vec3 rayOrigin, vec3 rayDir, vec3 sphereCenter, float radius, out vec3 intersectionPoint, out vec3 normal);

vec4 getSpeedColor(float speed, float minSpeed, float maxSpeed)
{
    float normalizedSpeed = clamp((speed - minSpeed) / (maxSpeed - minSpeed), 0.0, 1.0);

    const vec3 colorBlue = vec3(0.0, 0.0, 1.0);
    const vec3 colorGreen = vec3(0.0, 1.0, 0.0);
    const vec3 colorYellow = vec3(1.0, 1.0, 0.0);
    const vec3 colorOrange = vec3(1.0, 0.5, 0.0);
    const vec3 colorRed = vec3(1.0, 0.0, 0.0);

    vec3 finalColor;

    if (normalizedSpeed < 0.25) {
        float t = normalizedSpeed / 0.25;
        finalColor = mix(colorBlue, colorGreen, t);
    } else if (normalizedSpeed < 0.5) {
        float t = (normalizedSpeed - 0.25) / 0.25;
        finalColor = mix(colorGreen, colorYellow, t);
    } else if (normalizedSpeed < 0.75) {
        float t = (normalizedSpeed - 0.5) / 0.25;
        finalColor = mix(colorYellow, colorOrange, t);
    } else {
        float t = (normalizedSpeed - 0.75) / 0.25;
        finalColor = mix(colorOrange, colorRed, t);
    }

    return vec4(finalColor, 1.0);
}

vec4 getDensityDeviationColor(float deviation)
{
    vec3 color;

    if (deviation < -0.2) {
        color = vec3(0.0, 0.0, 1.0);
    } else if (deviation < -0.1) {
        float t = (deviation + 0.2) / 0.1;
        color = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), t);
    } else if (deviation < 0.1) {
        float t = (deviation + 0.1) / 0.2;
        color = mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), t);
    } else if (deviation < 0.2) {
        float t = (deviation - 0.1) / 0.1;
        color = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.5, 0.0), t);
    } else {
        color = vec3(1.0, 0.0, 0.0);
    }

    return vec4(color, 1.0);
}

void main() {
    vec2 _uv = uv * 2.0 - 1.0;

    if (dot(_uv, _uv) > 1.0) {
        discard;
    }

    vec3 cameraPosition = ubo.inverseView[3].xyz;
    vec3 rayOrigin = cameraPosition;
    vec3 rayDir = normalize(worldPosition - cameraPosition);

    vec3 intersectionPoint;
    vec3 normal;

    if (!raySphereIntersection(rayOrigin, rayDir, sphereCenter, radius, intersectionPoint, normal)) {
        discard;
    }

    vec3 totalLight = ubo.globalAmbient;

    for (uint i = 0; i < ubo.activeDirectionalLights; i++) {
        totalLight += calculateDirectionalLight(ubo.directionalLights[i], normal);
    }

    for (uint i = 0; i < ubo.activePointLights; i++) {
        totalLight += calculatePointLight(ubo.pointLights[i], normal, intersectionPoint);
    }

    for (uint i = 0; i < ubo.activeSpotLights; i++) {
        totalLight += calculateSpotLight(ubo.spotLights[i], normal, intersectionPoint);
    }

    //outColor = vec4(getSpeedColor(length(velocity), 0.F, 5.F)) * vec4(totalLight, 1.0);
    outColor = getDensityDeviationColor(densityDeviation) * vec4(totalLight, 1.0);
}

bool raySphereIntersection(vec3 rayOrigin, vec3 rayDir, vec3 sphereCenter, float radius, out vec3 intersectionPoint, out vec3 normal) {
    vec3 oc = rayOrigin - sphereCenter;
    float a = dot(rayDir, rayDir);
    float b = 2.0 * dot(oc, rayDir);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4.0 * a * c;

    if (discriminant < 0.0) {
        return false;
    }

    float t = (-b - sqrt(discriminant)) / (2.0 * a);

    if (t < 0.0) {
        t = (-b + sqrt(discriminant)) / (2.0 * a);
    }

    if (t < 0.0) {
        return false;
    }

    intersectionPoint = rayOrigin + t * rayDir;
    normal = normalize(intersectionPoint - sphereCenter);

    return true;
}

vec3 calculateLight(BaseLight light, vec3 lightDirection, vec3 normal, vec3 intersectionPoint) {
    float lambertian = max(dot(lightDirection, normal), 0.0);
    float specularValue = 0.0;

    if (lambertian > 0.0) {
        vec3 cameraWorldPosition = ubo.inverseView[3].xyz;
        vec3 viewDirection = normalize(cameraWorldPosition - intersectionPoint);
        vec3 halfDirection = normalize(lightDirection + viewDirection);
        float specularAngle = max(dot(halfDirection, normal), 0.0);
        specularValue = pow(specularAngle, 64.0);
    }

    vec3 ambient = light.ambient;
    vec3 diffuse = light.diffuse * lambertian * light.intensity;
    vec3 specular = light.specular * specularValue * light.intensity;

    return ambient + diffuse + specular;
}

vec3 calculatePointLight(PointLight light, vec3 normal, vec3 intersectionPoint) {
    vec3 lightDirection = light.position - intersectionPoint;
    float distance = length(lightDirection);
    lightDirection = normalize(lightDirection);

    float attenuation = light.attenuation.constant -
    light.attenuation.linear * distance -
    light.attenuation.exp * distance * distance;
    attenuation = max(attenuation, 0.0);

    return calculateLight(light.base, lightDirection, normal, intersectionPoint) * attenuation;
}

vec3 calculateDirectionalLight(DirectionalLight light, vec3 normal) {
    return calculateLight(light.base, normalize(light.direction), normal, worldPosition);
}

vec3 calculateSpotLight(SpotLight light, vec3 normal, vec3 intersectionPoint) {
    vec3 lightToPixel = normalize(intersectionPoint - light.base.position);
    float spotFactor = dot(lightToPixel, normalize(light.direction));

    if (spotFactor > light.cutOff) {
        vec3 color = calculatePointLight(light.base, normal, intersectionPoint);
        float spotLightIntensity = (1.0 - (1.0 - spotFactor)/(1.0 - light.cutOff));
        return color * spotLightIntensity;
    } else {
        return light.base.base.ambient;
    }
}
