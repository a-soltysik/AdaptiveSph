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
    // Normalize speed to [0,1] range
    float normalizedSpeed = clamp((speed - minSpeed) / (maxSpeed - minSpeed), 0.0, 1.0);

    // Define color keypoints (blue -> green -> yellow -> orange -> red)
    const vec3 colorBlue = vec3(0.0, 0.0, 1.0);
    const vec3 colorGreen = vec3(0.0, 1.0, 0.0);
    const vec3 colorYellow = vec3(1.0, 1.0, 0.0);
    const vec3 colorOrange = vec3(1.0, 0.5, 0.0);
    const vec3 colorRed = vec3(1.0, 0.0, 0.0);

    // Determine which segment of the gradient we're in
    vec3 finalColor;

    if (normalizedSpeed < 0.25) {
        // Blue to Green
        float t = normalizedSpeed / 0.25;
        finalColor = mix(colorBlue, colorGreen, t);
    } else if (normalizedSpeed < 0.5) {
        // Green to Yellow
        float t = (normalizedSpeed - 0.25) / 0.25;
        finalColor = mix(colorGreen, colorYellow, t);
    } else if (normalizedSpeed < 0.75) {
        // Yellow to Orange
        float t = (normalizedSpeed - 0.5) / 0.25;
        finalColor = mix(colorYellow, colorOrange, t);
    } else {
        // Orange to Red
        float t = (normalizedSpeed - 0.75) / 0.25;
        finalColor = mix(colorOrange, colorRed, t);
    }

    // Return final color with full opacity
    return vec4(finalColor, 1.0);
}

vec4 getDensityDeviationColor(float deviation)
{
    // Visualize density deviation with different colors
    vec3 color;

    if (deviation < -0.2) {
        color = vec3(0.0, 0.0, 1.0);// Blue for very low density (<-20%)
    } else if (deviation < -0.1) {
        float t = (deviation + 0.2) / 0.1;// Normalize to [0,1]
        color = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), t);// Blue to Cyan
    } else if (deviation < 0.1) {
        float t = (deviation + 0.1) / 0.2;// Normalize to [0,1]
        color = mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), t);// Cyan to Green
    } else if (deviation < 0.2) {
        float t = (deviation - 0.1) / 0.1;// Normalize to [0,1]
        color = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.5, 0.0), t);// Green to Orange
    } else {
        color = vec3(1.0, 0.0, 0.0);// Red for very high density (>20%)
    }

    return vec4(color, 1.0);
}

vec4 getSpeedTurbo(float speed, float minSpeed, float maxSpeed)
{
    float t = clamp((speed - minSpeed) / (maxSpeed - minSpeed), 0.0, 1.0);

    // Aproksymacja Turbo colormap
    const vec4 kRedVec4 = vec4(0.55305649, 3.00913185, -5.46192616, -11.11819092);
    const vec4 kGreenVec4 = vec4(0.16207513, 0.17712472, 15.24091500, -36.50657960);
    const vec4 kBlueVec4 = vec4(-0.05195877, 5.18000081, -30.94853351, 81.96403246);
    const vec2 kRedVec2 = vec2(27.81927491, -14.87899417);
    const vec2 kGreenVec2 = vec2(25.95549545, -5.02738237);
    const vec2 kBlueVec2 = vec2(-86.53476570, 30.23299484);

    // Obliczenia dla każdego kanału
    float r = kRedVec4.x + kRedVec4.y * t + kRedVec4.z * t * t + kRedVec4.w * t * t * t + kRedVec2.x * t * t * t * t + kRedVec2.y * t * t * t * t * t;
    float g = kGreenVec4.x + kGreenVec4.y * t + kGreenVec4.z * t * t + kGreenVec4.w * t * t * t + kGreenVec2.x * t * t * t * t + kGreenVec2.y * t * t * t * t * t;
    float b = kBlueVec4.x + kBlueVec4.y * t + kBlueVec4.z * t * t + kBlueVec4.w * t * t * t + kBlueVec2.x * t * t * t * t + kBlueVec2.y * t * t * t * t * t;

    return vec4(clamp(r, 0.0, 1.0), clamp(g, 0.0, 1.0), clamp(b, 0.0, 1.0), 1.0);
}

vec4 getSpeedNeon(float speed, float minSpeed, float maxSpeed)
{
    float normalizedSpeed = clamp((speed - minSpeed) / (maxSpeed - minSpeed), 0.0, 1.0);

    // Kolory neonowe
    vec3 neonColors[5] = vec3[](
    vec3(0.1, 0.0, 0.3), // Ciemny fiolet
    vec3(0.0, 1.0, 1.0), // Cyjan
    vec3(0.0, 1.0, 0.0), // Zielony
    vec3(1.0, 1.0, 0.0), // Żółty
    vec3(1.0, 0.0, 1.0)// Magenta
    );

    // Znajdź segment
    float scaledSpeed = normalizedSpeed * 4.0;// 0-4 range
    int index = int(floor(scaledSpeed));
    float fraction = scaledSpeed - float(index);

    // Clamp index
    index = clamp(index, 0, 3);

    // Interpoluj między kolorami
    vec3 color1 = neonColors[index];
    vec3 color2 = neonColors[index + 1];
    vec3 finalColor = mix(color1, color2, smoothstep(0.0, 1.0, fraction));

    // Dodaj neonowy glow effect
    float glow = 1.0 + normalizedSpeed * 0.5;
    finalColor *= glow;

    return vec4(finalColor, 1.0);
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

    outColor = getSpeedNeon(length(velocity), 0.0, 3.0) * vec4(totalLight, 1.0);
    //outColor = getDensityDeviationColor(densityDeviation) * vec4(totalLight, 1.0);
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
