#version 450
#extension GL_GOOGLE_include_directive : require
#include "scene.glsl"

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;
layout(location = 0) out vec3 vertex_color;

void main() {
    gl_Position = scene.view_projection * vec4(in_position, 1.0);
    vertex_color = in_color;
}
