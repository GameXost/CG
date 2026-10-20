layout(set = 0, binding = 0, std140) uniform SceneData {
    mat4 view_projection;
    vec4 camera_position;
    vec4 camera_direction; // W: 0 — перспектива, 1 — ортографическая проекция
    vec4 ambient;
    vec4 light_direction;
    vec4 light_color;
    ivec4 light_counts; // X — point, Y — spot
} scene;
