#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

layout(set = 0, binding = 0, std140) uniform ObjectData {
    mat4 mvp;
    vec4 tint; // RGB умножает цвет вершин, W включает процедурную окраску
} object;

layout(location = 0) out vec3 vertex_color;

void main() {
    gl_Position = object.mvp * vec4(in_position, 1.0);
    vertex_color = mix(vec3(1.0), in_color, object.tint.w) * object.tint.rgb;
}
