#version 450
#extension GL_GOOGLE_include_directive : require
#include "scene.glsl"
#include "model.glsl"

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;
layout(location = 2) in vec3 in_normal;
layout(location = 3) in vec2 in_uv;

layout(location = 0) out vec3 world_position;
layout(location = 1) out vec3 world_normal;
layout(location = 2) out vec3 vertex_color;
layout(location = 3) out vec2 texture_uv;

void main() {
    vec4 position = object.model * vec4(in_position, 1.0);
    gl_Position = scene.view_projection * position;
    world_position = position.xyz;
    world_normal = mat3(object.normal_matrix) * in_normal;
    vertex_color = mix(vec3(1.0), in_color, object.albedo.w) * object.albedo.rgb;
    texture_uv = in_uv;
}
