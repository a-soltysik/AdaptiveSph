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

layout (push_constant) uniform Push {
    vec3 translation;
    vec3 scale;
    vec3 rotation;
} push;

void main() {
    mat3 rotationMatrix = quatToMat3(eulerToQuat(push.rotation));
    mat4 modelMatrix = mat4(
    vec4(rotationMatrix[0] * push.scale.x, 0.0),
    vec4(rotationMatrix[1] * push.scale.y, 0.0),
    vec4(rotationMatrix[2] * push.scale.z, 0.0),
    vec4(push.translation, 1.0)
    );

    vec4 worldPosition = modelMatrix * vec4(position, 1.0);
    gl_Position = ubo.projection * (ubo.view * worldPosition);

    fragNormalWorld = normalize(rotationMatrix * normal);
    fragWorldPosition = worldPosition.xyz;
    fragTexCoord = uv;
}
