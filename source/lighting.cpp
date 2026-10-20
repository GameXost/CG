#include "lighting.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <numbers>

namespace lighting {
namespace {

void sanitizeValue(float& value, float fallback) {
    if (!std::isfinite(value)) value = fallback;
}

void sanitizeValue(glm::vec3& value, glm::vec3 fallback) {
    for (int i = 0; i < 3; ++i) sanitizeValue(value[i], fallback[i]);
}

} // namespace

void reset(State& state) {
    state = {};
    state.point_lights = {{{-2.5f, 2.5f, 2.0f}, {1.0f, 0.55f, 0.25f}, 6.0f}};
    state.spot_lights = {{{2.5f, 3.0f, 2.5f}, {-1.0f, -3.0f, -2.5f},
                          {0.35f, 0.65f, 1.0f}, 15.0f, 15.0f, 28.0f}};
    state.spot_lights[0].direction = lightDirection(state.spot_lights[0].direction);
}

void sanitize(State& state) {
    // Ручной ввод ImGui допускает NaN.
    sanitizeValue(state.ambient_color, glm::vec3(1.0f));
    state.ambient_color = glm::clamp(state.ambient_color, 0.0f, 1.0f);
    sanitizeValue(state.ambient_intensity, 0.12f);
    sanitizeValue(state.directional_color, glm::vec3(1.0f));
    state.directional_color = glm::clamp(state.directional_color, 0.0f, 1.0f);
    sanitizeValue(state.directional_intensity, 0.7f);
    sanitizeValue(state.directional_direction, {0.0f, -1.0f, 0.0f});

    const PointLight point_defaults;
    for (auto& light : state.point_lights) {
        sanitizeValue(light.position, point_defaults.position);
        sanitizeValue(light.color, point_defaults.color);
        light.color = glm::clamp(light.color, 0.0f, 1.0f);
        sanitizeValue(light.intensity, point_defaults.intensity);
    }
    const SpotLight spot_defaults;
    for (auto& light : state.spot_lights) {
        sanitizeValue(light.position, spot_defaults.position);
        sanitizeValue(light.direction, {0.0f, -1.0f, 0.0f});
        sanitizeValue(light.color, spot_defaults.color);
        light.color = glm::clamp(light.color, 0.0f, 1.0f);
        sanitizeValue(light.intensity, spot_defaults.intensity);
        sanitizeValue(light.inner_angle, spot_defaults.inner_angle);
        sanitizeValue(light.outer_angle, spot_defaults.outer_angle);
        light.outer_angle = std::max(light.outer_angle, light.inner_angle);
    }
}

void drawInterface(State& state, uint64_t max_storage_range) {
    constexpr auto clamp = ImGuiSliderFlags_AlwaysClamp;
    if (ImGui::CollapsingHeader("Ambient and directional", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Ambient color", &state.ambient_color.x);
        ImGui::SliderFloat("Ambient intensity", &state.ambient_intensity, 0.0f, 1.0f, "%.2f", clamp);
        ImGui::ColorEdit3("Light color", &state.directional_color.x);
        ImGui::SliderFloat("Light intensity", &state.directional_intensity, 0.0f, 3.0f, "%.2f", clamp);
        ImGui::SliderFloat3("Direction", &state.directional_direction.x, -1.0f, 1.0f, "%.2f", clamp);
        ImGui::TextWrapped("Direction follows the light rays; zero uses (0, -1, 0).");
    }

    ImGui::Checkbox("Show light markers", &state.show_markers);
    if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::BeginDisabled((state.point_lights.size() + 1) * sizeof(PointData) > max_storage_range);
        if (ImGui::Button("Add point")) state.point_lights.push_back({});
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::Text("Count: %zu", state.point_lights.size());
        size_t remove_index = state.point_lights.size();
        for (size_t i = 0; i < state.point_lights.size(); ++i) {
            ImGui::PushID(int(i));
            if (ImGui::TreeNode("Point", "Point %zu", i + 1)) {
                auto& light = state.point_lights[i];
                ImGui::SliderFloat3("Position", &light.position.x, -10.0f, 10.0f, "%.2f", clamp);
                ImGui::ColorEdit3("Color", &light.color.x);
                ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 100.0f, "%.2f", clamp);
                if (ImGui::Button("Remove point")) remove_index = i;
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
        // Удаляем после обхода, чтобы не повторять ImGui ID.
        if (remove_index < state.point_lights.size()) state.point_lights.erase(state.point_lights.begin() + remove_index);
    }
    if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::BeginDisabled((state.spot_lights.size() + 1) * sizeof(SpotData) > max_storage_range);
        if (ImGui::Button("Add spot")) state.spot_lights.push_back({});
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::Text("Count: %zu", state.spot_lights.size());
        size_t remove_index = state.spot_lights.size();
        for (size_t i = 0; i < state.spot_lights.size(); ++i) {
            ImGui::PushID(int(i));
            if (ImGui::TreeNode("Spot", "Spot %zu", i + 1)) {
                auto& light = state.spot_lights[i];
                ImGui::SliderFloat3("Position", &light.position.x, -10.0f, 10.0f, "%.2f", clamp);
                ImGui::SliderFloat3("Direction", &light.direction.x, -1.0f, 1.0f, "%.2f", clamp);
                ImGui::ColorEdit3("Color", &light.color.x);
                ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 100.0f, "%.2f", clamp);
                ImGui::SliderFloat("Inner angle", &light.inner_angle, 1.0f, 89.0f, "%.1f deg", clamp);
                sanitizeValue(light.inner_angle, SpotLight{}.inner_angle);
                light.outer_angle = std::max(light.outer_angle, light.inner_angle);
                ImGui::SliderFloat("Outer angle", &light.outer_angle, light.inner_angle, 89.0f, "%.1f deg", clamp);
                ImGui::TextUnformatted("Angles are cone half-angles.");
                if (ImGui::Button("Remove spot")) remove_index = i;
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
        if (remove_index < state.spot_lights.size()) state.spot_lights.erase(state.spot_lights.begin() + remove_index);
    }
}

glm::vec3 lightDirection(glm::vec3 direction) {
    const float length_squared = glm::dot(direction, direction);
    if (!std::isfinite(length_squared) || length_squared <= 1.0e-12f)
        return {0.0f, -1.0f, 0.0f};
    return direction / std::sqrt(length_squared);
}

MarkerMesh makeLightMarkers(const std::vector<PointLight>& points,
                            const std::vector<SpotLight>& spots) {
    constexpr uint32_t ring_segments = 32;
    constexpr float half_extent = 0.12f, cone_length = 1.2f;
    MarkerMesh mesh;
    mesh.vertices.reserve(36 * (points.size() + spots.size()) + 138 * spots.size());

    const glm::vec3 corners[] = {
        {-1.0f, -1.0f, -1.0f}, {1.0f, -1.0f, -1.0f},
        {1.0f, 1.0f, -1.0f}, {-1.0f, 1.0f, -1.0f},
        {-1.0f, -1.0f, 1.0f}, {1.0f, -1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f}, {-1.0f, 1.0f, 1.0f},
    };
    const uint32_t indices[] = {
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
        0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5,
        0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2,
    };
    auto addCube = [&](glm::vec3 position, glm::vec3 color) {
        for (uint32_t index : indices)
            mesh.vertices.push_back({position + half_extent * corners[index], color});
    };
    for (const auto& light : points) addCube(light.position, light.color);
    for (const auto& light : spots) addCube(light.position, light.color);
    mesh.triangle_vertices = uint32_t(mesh.vertices.size());

    auto addLine = [&](glm::vec3 from, glm::vec3 to, glm::vec3 color) {
        mesh.vertices.push_back({from, color});
        mesh.vertices.push_back({to, color});
    };
    for (const auto& light : spots) {
        const glm::vec3 direction = lightDirection(light.direction);
        // Избегаем параллельных осей при построении базиса.
        const glm::vec3 reference = std::abs(direction.y) > 0.99f ?
            glm::vec3(1.0f, 0.0f, 0.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
        const glm::vec3 right = glm::normalize(glm::cross(direction, reference));
        const glm::vec3 up = glm::cross(right, direction);
        const glm::vec3 center = light.position + cone_length * direction;
        const float inner_radius = cone_length * std::tan(glm::radians(light.inner_angle));
        const float outer_radius = cone_length * std::tan(glm::radians(light.outer_angle));
        auto ringPoint = [&](uint32_t index, float radius) {
            const float angle = 2.0f * std::numbers::pi_v<float> * float(index) / float(ring_segments);
            return center + radius * (std::cos(angle) * right + std::sin(angle) * up);
        };
        for (uint32_t i = 0; i < ring_segments; ++i)
            addLine(ringPoint(i, inner_radius), ringPoint((i + 1) % ring_segments, inner_radius),
                    glm::vec3(0.45f));
        for (uint32_t i = 0; i < ring_segments; ++i)
            addLine(ringPoint(i, outer_radius), ringPoint((i + 1) % ring_segments, outer_radius),
                    glm::vec3(1.0f));
        for (uint32_t i = 0; i < 4; ++i)
            addLine(light.position, ringPoint(i * ring_segments / 4, outer_radius), glm::vec3(1.0f));
        addLine(light.position, center, glm::vec3(1.0f));
    }
    return mesh;
}

} // namespace lighting
