#version 450

const vec2 OFFSETS[6] = vec2[](
vec2(-1.0, -1.0),
vec2(-1.0, 1.0),
vec2(1.0, -1.0),
vec2(1.0, -1.0),
vec2(-1.0, 1.0),
vec2(1.0, 1.0)
);

layout (location = 0) out vec2 uv;
layout (location = 1) out vec3 worldPosition;
layout (location = 2) out vec3 sphereCenter;
layout (location = 3) out float radius;
layout (location = 4) out vec3 color;

layout (set = 0, binding = 0) uniform VertUbo
{
    mat4 projection;
    mat4 view;
} ubo;

layout (set = 0, binding = 2) readonly buffer PositionBuffer {
    vec4 positions[];
};

layout (set = 0, binding = 3) readonly buffer ColorBuffer {
    vec4 colors[];
};

layout (set = 0, binding = 4) readonly buffer RadiusBuffer {
    float radiuses[];
};

void main() {
    vec2 quadCoord = OFFSETS[gl_VertexIndex];

    vec3 viewRight = normalize(vec3(ubo.view[0][0], ubo.view[1][0], ubo.view[2][0]));
    vec3 viewUp = normalize(vec3(ubo.view[0][1], ubo.view[1][1], ubo.view[2][1]));

    float scale = 2 * radiuses[gl_InstanceIndex];

    vec3 vertexPosition = positions[gl_InstanceIndex].xyz;
    vertexPosition += (quadCoord.x * 0.5) * viewRight * scale;
    vertexPosition += (quadCoord.y * 0.5) * viewUp * scale;

    uv = (quadCoord + vec2(1.0)) * 0.5;
    radius = radiuses[gl_InstanceIndex];
    worldPosition = vertexPosition;
    sphereCenter = positions[gl_InstanceIndex].xyz;
    color = colors[gl_InstanceIndex].xyz;

    gl_Position = ubo.projection * ubo.view * vec4(vertexPosition, 1.0);
}
