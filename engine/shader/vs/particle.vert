#version 450

#include "utils.glsl"

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;

layout (location = 0) out vec3 fragWorldPosition;
layout (location = 1) out vec3 fragNormalWorld;
layout (location = 2) out vec2 fragTexCoord;

layout (set = 0, binding = 0) uniform VertUbo
{
    mat4 projection;
    mat4 view;
} ubo;

struct InstanceData {
    vec3 position;
    vec3 velocity;
    vec3 force;
    float density;
    float pressure;
    float mass;
};

layout (set = 0, binding = 3) readonly buffer InstanceBuffer {
    InstanceData instances[];
};

void main() {
    InstanceData instance = instances[gl_InstanceIndex];

    float scale = instance.mass;

    mat4 modelMatrix = mat4(
    scale, 0.0, 0.0, 0.0,
    0.0, scale, 0.0, 0.0,
    0.0, 0.0, scale, 0.0,
    instance.position.x, instance.position.y, instance.position.z, 1.0
    );

    vec4 worldPosition = modelMatrix * vec4(position, 1.0);
    gl_Position = ubo.projection * (ubo.view * worldPosition);

    fragNormalWorld = normal;
    fragWorldPosition = worldPosition.xyz;
    fragTexCoord = uv;
}
