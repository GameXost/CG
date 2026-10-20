#pragma once

#include <glm/glm.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lighting {

struct PointLight {
    glm::vec3 position{0.0f, 2.0f, 2.0f};
    glm::vec3 color{1.0f};
    float intensity = 8.0f;
};

struct SpotLight {
    glm::vec3 position{0.0f, 3.0f, 3.0f};
    glm::vec3 direction{0.0f, -1.0f, -1.0f};
    glm::vec3 color{1.0f};
    float intensity = 16.0f;
    float inner_angle = 15.0f;
    float outer_angle = 25.0f;
};

struct State {
    glm::vec3 ambient_color{1.0f}, directional_color{1.0f};
    glm::vec3 directional_direction{-0.5f, -1.0f, -0.6f};
    float ambient_intensity = 0.12f, directional_intensity = 0.7f;
    std::vector<PointLight> point_lights;
    std::vector<SpotLight> spot_lights;
    bool show_markers = true;
};

// Поля и смещения совпадают с std430 в шейдере.
struct alignas(16) PointData {
    glm::vec4 position, color;
};
struct alignas(16) SpotData {
    glm::vec4 position, direction, color, cutoff;
};
static_assert(sizeof(PointData) == 32 && sizeof(SpotData) == 64);
static_assert(offsetof(PointData, color) == 16 && offsetof(SpotData, cutoff) == 48);

struct MarkerVertex {
    glm::vec3 position;
    glm::vec3 color;
};
static_assert(sizeof(MarkerVertex) == 24 && offsetof(MarkerVertex, color) == 12);

struct MarkerMesh {
    std::vector<MarkerVertex> vertices;
    uint32_t triangle_vertices = 0;
};

void reset(State& state);
void sanitize(State& state);
void drawInterface(State& state, uint64_t max_storage_range);
glm::vec3 lightDirection(glm::vec3 direction);
MarkerMesh makeLightMarkers(const std::vector<PointLight>& points,
                            const std::vector<SpotLight>& spots);

} // namespace lighting
