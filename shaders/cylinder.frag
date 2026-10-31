#version 450
#extension GL_GOOGLE_include_directive : require
#include "scene.glsl"
#include "model.glsl"

layout(set = 1, binding = 1) uniform sampler2D albedo_texture;
layout(push_constant) uniform SamplingParameters {
    float time;
    float strength;
    float speed;
    uint enabled;
} sampling;

struct PointLight {
    vec4 position;
    vec4 color; // RGB — цвет, W — интенсивность
};
struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 color;
    vec4 cutoff; // cos(inner), cos(outer)
};
layout(set = 0, binding = 1, std430) readonly buffer PointLights {
    PointLight points[];
};
layout(set = 0, binding = 2, std430) readonly buffer SpotLights {
    SpotLight spots[];
};

layout(location = 0) in vec3 world_position;
layout(location = 1) in vec3 world_normal;
layout(location = 2) in vec3 vertex_color;
layout(location = 3) in vec2 texture_uv;
layout(location = 0) out vec4 out_color;

vec3 safeNormalize(vec3 value) {
    return value * inversesqrt(max(dot(value, value), 1e-12));
}

vec3 lighting(vec3 n, vec3 v, vec3 l, vec3 intensity, vec3 albedo) {
    float diffuse = max(dot(n, l), 0.0);
    if (diffuse <= 0.0) return vec3(0.0);
    vec3 h = safeNormalize(l + v);
    float specular = pow(max(dot(n, h), 0.0), object.specular.w);
    return intensity * (albedo * diffuse + object.specular.rgb * specular);
}

void main() {
    float phase = sampling.time * sampling.speed;
    vec2 uv = texture_uv;
    if (sampling.enabled != 0u) {
        uv += sampling.strength * vec2(0.12 * sin(phase), 0.08 * cos(phase * 0.7));
    }
    vec3 albedo = texture(albedo_texture, uv).rgb;
    if (sampling.enabled != 0u) {
        vec2 second_uv = texture_uv * 1.7
            + sampling.strength * vec2(-0.1 * phase, 0.13 * sin(phase * 0.8));
        albedo = mix(albedo, texture(albedo_texture, second_uv).rgb, 0.5 * sampling.strength);
        vec3 modulation = 0.65 + 0.35 * sin(phase + vec3(0.0, 2.094, 4.189));
        albedo *= mix(vec3(1.0), modulation, sampling.strength);
    }
    albedo *= vertex_color;
    vec3 n = safeNormalize(world_normal);
    vec3 v = scene.camera_direction.w == 0.0
        ? safeNormalize(scene.camera_position.xyz - world_position)
        : -scene.camera_direction.xyz;
    vec3 color = scene.ambient.rgb * albedo;
    color += lighting(n, v, -scene.light_direction.xyz, scene.light_color.rgb, albedo);

    for (int i = 0; i < scene.light_counts.x; ++i) {
        vec3 delta = points[i].position.xyz - world_position;
        // Защита от деления на ноль.
        float distance_squared = max(dot(delta, delta), 1e-4);
        color += lighting(n, v, safeNormalize(delta),
                          points[i].color.rgb * points[i].color.w / distance_squared, albedo);
    }
    for (int i = 0; i < scene.light_counts.y; ++i) {
        vec3 delta = spots[i].position.xyz - world_position;
        vec3 l = safeNormalize(delta);
        float cosine = dot(-l, spots[i].direction.xyz);
        float inner = spots[i].cutoff.x, outer = spots[i].cutoff.y;
        float cone = inner > outer ? clamp((cosine - outer) / (inner - outer), 0.0, 1.0)
                                   : step(inner, cosine);
        float distance_squared = max(dot(delta, delta), 1e-4);
        color += lighting(n, v, l, spots[i].color.rgb * spots[i].color.w * cone / distance_squared,
                          albedo);
    }
    out_color = vec4(color, 1.0);
}
