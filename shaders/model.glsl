layout(set = 1, binding = 0, std140) uniform ModelData {
    mat4 model;
    mat4 normal_matrix;
    vec4 albedo; // W включает процедурную окраску вершин
    vec4 specular; // W — показатель блеска
} object;
