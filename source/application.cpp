#include "application.hpp"
#include "lighting.hpp"
#include "gpu_resources.hpp"

#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace application {
namespace {

// Геометрия и размеры интерфейса
constexpr uint32_t segments = 50;
constexpr uint32_t cylinder_count = 2;
constexpr uint32_t floor_object = cylinder_count;
constexpr uint32_t object_count = cylinder_count + 1;
constexpr float panel_width = 340.0f;
constexpr float scene_left = panel_width + 24.0f;
constexpr float default_ortho_height = 6.0f;

// Начальные настройки анимации
constexpr float default_anim_speed = 1.0f;
constexpr float default_path_radius = 0.8f;
constexpr float default_path_height = 0.4f;
constexpr float default_spin_speed = 45.0f;

// Геометрия и данные объектов
struct Vertex {
    glm::vec3 position;
    glm::vec3 color;
    glm::vec3 normal;
    glm::vec2 uv;
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

struct Camera {
    glm::vec3 position{3.617f, 2.781f, 7.758f};
    float yaw = -155.0f;
    float pitch = -18.0f;
    float fov = 45.0f;
};

using resources::Buffer;
using resources::createBuffer;
using resources::writeBuffer;
using resources::destroyBuffer;

// Раскладка std140 для UBO сцены и модели.
struct alignas(16) SceneData {
    glm::mat4 view_projection;
    glm::vec4 camera_position;
    glm::vec4 camera_direction;
    glm::vec4 ambient;
    glm::vec4 light_direction;
    glm::vec4 light_color;
    glm::ivec4 light_counts;
};
struct alignas(16) ModelData {
    glm::mat4 model;
    glm::mat4 normal_matrix;
    glm::vec4 albedo;
    glm::vec4 specular;
};
using lighting::PointData;
using lighting::SpotData;
static_assert(sizeof(SceneData) == 160 && offsetof(SceneData, light_counts) == 144);
static_assert(sizeof(ModelData) == 160 && offsetof(ModelData, specular) == 144);
static_assert(offsetof(Vertex, normal) == 24);
static_assert(offsetof(Vertex, uv) == 36 && sizeof(Vertex) == 44);
static_assert(offsetof(Vertex, position) == offsetof(lighting::MarkerVertex, position));
static_assert(offsetof(Vertex, color) == offsetof(lighting::MarkerVertex, color));

struct Object {
    glm::vec3 position{0.0f};
    glm::vec3 rotation{0.0f}; // Углы в градусах
    glm::vec3 scale{1.0f};
    glm::vec3 color{1.0f};
    glm::vec3 specular{0.5f};
    float shininess = 32.0f;
    bool visible = true;
    bool procedural_color = false;
    int material = 0;
    bool sampling_effect = false;
    float sampling_strength = 0.65f;
    float sampling_speed = 1.0f;
};

struct Material {
    resources::Texture albedo;
    VkDescriptorSet descriptor = VK_NULL_HANDLE;
};

constexpr uint32_t material_count = 3;
struct MaterialFiles {
    const char* name;
    const char* albedo;
};
constexpr std::array<MaterialFiles, material_count> material_files{{
    {"Alucard", "assets/textures/alucard_albedo.png"},
    {"Space Cat", "assets/textures/space_cat_albedo.png"},
    {"Cupcake Cat", "assets/textures/cupcake_cat_albedo.png"},
}};
struct SamplingParameters {
    float time, strength, speed;
    uint32_t enabled;
};
static_assert(sizeof(SamplingParameters) == 16 && offsetof(SamplingParameters, enabled) == 12);

// Ресурсы Vulkan
Buffer vertex_buffer, index_buffer;
Buffer scene_buffer, model_buffer, point_buffer, spot_buffer, marker_buffer, marker_staging;
VkDeviceSize model_stride = 0;
VkDeviceSize marker_bytes = 0;
uint32_t index_count = 0;
std::array<VkDescriptorSetLayout, 2> descriptor_layouts{};
VkDescriptorSet scene_descriptor = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;
VkPipeline marker_pipeline = VK_NULL_HANDLE, line_pipeline = VK_NULL_HANDLE;
std::array<Material, material_count> materials;
VkSampler texture_sampler = VK_NULL_HANDLE;
VkDeviceSize max_storage_range = 0;

// Параметры сцены и камеры
std::array<Object, object_count> objects;
int selected_object = 0;
int projection_type = 0;
Camera camera;
float ortho_height = default_ortho_height;
lighting::State lights;
uint32_t marker_triangle_count = 0, marker_line_count = 0;

// Состояние и параметры анимации
bool animate = false;
bool playing = true;
float animation_speed = default_anim_speed;
float path_radius = default_path_radius;
float path_height = default_path_height;
float spin_speed = default_spin_speed;
double animation_time = 0.0;
double previous_time = -1.0;
double sampling_time = 0.0;

// Цилиндр вдоль Y: радиус 1, высота 2
Mesh makeCylinder() {
    Mesh mesh;
    mesh.vertices.reserve(4 * segments + 4);
    mesh.indices.reserve(12 * segments);

    auto addVertex = [&](glm::vec3 position, glm::vec3 normal, glm::vec2 uv) {
        mesh.vertices.push_back({position, (position + glm::vec3(1.0f)) * 0.5f, normal, uv});
    };

    // На шве совпадают позиции, но UV различаются: U=0 и U=1.
    for (uint32_t i = 0; i <= segments; ++i) {
        const float u = float(i) / float(segments);
        const float angle = i == segments ? 0.0f : 2.0f * std::numbers::pi_v<float> * u;
        const float x = std::cos(angle), z = std::sin(angle);
        addVertex({x, -1.0f, z}, {x, 0.0f, z}, {u, 1.0f});
        addVertex({x, 1.0f, z}, {x, 0.0f, z}, {u, 0.0f});
    }

    // Отдельные вершины сохраняют резкую границу нормалей.
    const uint32_t cap_offset = uint32_t(mesh.vertices.size());
    for (uint32_t i = 0; i < segments; ++i) {
        const auto position = mesh.vertices[2 * i].position;
        const glm::vec2 uv = glm::vec2(position.x, position.z) * 0.5f + 0.5f;
        addVertex(position, {0.0f, -1.0f, 0.0f}, uv);
        addVertex(mesh.vertices[2 * i + 1].position, {0.0f, 1.0f, 0.0f}, uv);
    }
    const uint32_t bottom_center = uint32_t(mesh.vertices.size());
    addVertex({0.0f, -1.0f, 0.0f}, {0.0f, -1.0f, 0.0f}, {0.5f, 0.5f});
    const uint32_t top_center = uint32_t(mesh.vertices.size());
    addVertex({0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.5f, 0.5f});

    for (uint32_t i = 0; i < segments; ++i) {
        const uint32_t next = (i + 1) % segments;
        const uint32_t bottom = 2 * i, top = bottom + 1;
        const uint32_t next_bottom = 2 * (i + 1), next_top = next_bottom + 1;
        // Обход треугольников наружу.
        mesh.indices.insert(mesh.indices.end(), {
            bottom, top, next_bottom,
            top, next_top, next_bottom,
            bottom_center, cap_offset + bottom, cap_offset + 2 * next,
            top_center, cap_offset + 2 * next + 1, cap_offset + top,
        });
    }
    return mesh;
}

Mesh makeFloor() {
    return {
        {
            {{-8.0f, 0.0f, -8.0f}, glm::vec3(1.0f), {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
            {{-8.0f, 0.0f,  8.0f}, glm::vec3(1.0f), {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
            {{ 8.0f, 0.0f,  8.0f}, glm::vec3(1.0f), {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
            {{ 8.0f, 0.0f, -8.0f}, glm::vec3(1.0f), {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
        },
        {0, 1, 2, 0, 2, 3},
    };
}

glm::vec3 forward(const Camera& camera) {
    const float yaw = glm::radians(camera.yaw), pitch = glm::radians(camera.pitch);
    return {std::cos(pitch) * std::sin(yaw), std::sin(pitch),
            std::cos(pitch) * std::cos(yaw)};
}

glm::mat4 viewMatrix(const Camera& camera) {
    return glm::lookAtRH(camera.position, camera.position + forward(camera),
                         {0.0f, 1.0f, 0.0f});
}

glm::mat4 projectionMatrix(const Camera& camera, float aspect,
                           int projection_type, float ortho_height) {
    glm::mat4 projection;
    if (projection_type == 0) {
        projection = glm::perspectiveRH_ZO(glm::radians(camera.fov), aspect, 0.1f, 100.0f);
    } else {
        const float half_height = ortho_height * 0.5f;
        projection = glm::orthoRH_ZO(-half_height * aspect, half_height * aspect,
                                    -half_height, half_height, 0.1f, 100.0f);
    }
    projection[1][1] *= -1.0f;
    return projection;
}

bool check(VkResult result, const char* operation) {
    if (result == VK_SUCCESS) return true;
    std::cerr << operation << " failed (VkResult " << result << ")\n";
    return false;
}

void resetScene() {
    objects = {};
    objects[0].position = {-1.5f, 0.0f, 0.0f};
    objects[0].rotation = {12.0f, 120.0f, -10.0f};
    objects[0].material = 0;
    objects[1].position = {1.5f, 0.0f, 0.0f};
    objects[1].rotation = {-15.0f, 100.0f, 15.0f};
    objects[1].scale = glm::vec3(0.8f);
    objects[1].material = 1;
    objects[1].sampling_effect = true;
    objects[1].sampling_strength = 0.3f;
    objects[floor_object].position = {0.0f, -1.5f, 0.0f};
    objects[floor_object].material = 2;
    objects[floor_object].specular = glm::vec3(0.15f);

    selected_object = 0;
    projection_type = 0;
    camera = {};
    ortho_height = default_ortho_height;
    lighting::reset(lights);

    animate = false;
    playing = true;
    animation_speed = default_anim_speed;
    path_radius = default_path_radius;
    path_height = default_path_height;
    spin_speed = default_spin_speed;
    animation_time = 0.0;
    previous_time = -1.0;
    sampling_time = 0.0;
}

void sanitize(float& value, float fallback) {
    if (!std::isfinite(value)) value = fallback;
}

void sanitize(glm::vec3& value, glm::vec3 fallback) {
    for (int i = 0; i < 3; ++i) sanitize(value[i], fallback[i]);
}

void sanitizeScene() {
    // Ручной ввод ImGui допускает NaN.
    const Camera camera_defaults;
    sanitize(camera.position, camera_defaults.position);
    sanitize(camera.yaw, camera_defaults.yaw);
    sanitize(camera.pitch, camera_defaults.pitch);
    sanitize(camera.fov, camera_defaults.fov);
    sanitize(ortho_height, default_ortho_height);

    const Object object_defaults;
    for (auto& object : objects) {
        sanitize(object.position, object_defaults.position);
        sanitize(object.rotation, object_defaults.rotation);
        sanitize(object.scale, object_defaults.scale);
        sanitize(object.color, object_defaults.color);
        object.color = glm::clamp(object.color, 0.0f, 1.0f);
        sanitize(object.specular, object_defaults.specular);
        object.specular = glm::clamp(object.specular, 0.0f, 1.0f);
        sanitize(object.shininess, object_defaults.shininess);
        object.material = std::clamp(object.material, 0, int(material_count) - 1);
        sanitize(object.sampling_strength, object_defaults.sampling_strength);
        object.sampling_strength = std::clamp(object.sampling_strength, 0.0f, 1.0f);
        sanitize(object.sampling_speed, object_defaults.sampling_speed);
        object.sampling_speed = std::clamp(object.sampling_speed, 0.0f, 3.0f);
    }
    lighting::sanitize(lights);
    sanitize(animation_speed, default_anim_speed);
    sanitize(path_radius, default_path_radius);
    sanitize(path_height, default_path_height);
    sanitize(spin_speed, default_spin_speed);
}

// Привязка данных к шейдерам
void bindBuffer(VkDescriptorSet set, uint32_t binding, VkDescriptorType type,
                const Buffer& buffer, VkDeviceSize range = VK_WHOLE_SIZE) {
    const VkDescriptorBufferInfo info = {buffer.handle, 0,
        range == VK_WHOLE_SIZE ? buffer.capacity : range};
    const VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set, .dstBinding = binding, .descriptorCount = 1,
        .descriptorType = type, .pBufferInfo = &info,
    };
    vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
}

// Вызывать после ожидания fence.
void ensureBuffer(Buffer& buffer, VkDeviceSize bytes, VkBufferUsageFlags usage,
                  bool device_local = false) {
    if (buffer.capacity >= bytes) return;
    Buffer replacement;
    if (!createBuffer(bytes, usage, replacement, device_local))
        throw std::runtime_error("Cannot grow scene buffer");
    destroyBuffer(buffer);
    buffer = replacement;
}

bool loadMaterials() {
    const VkSamplerCreateInfo sampler = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .maxLod = 0.0f,
    };
    if (!check(vkCreateSampler(graphics::internal::context.device, &sampler,
                                nullptr, &texture_sampler), "vkCreateSampler")) return false;
    for (size_t i = 0; i < materials.size(); ++i) {
        auto& material = materials[i];
        const auto& files = material_files[i];
        if (!resources::loadTexture(files.albedo, VK_FORMAT_R8G8B8A8_SRGB, material.albedo))
            return false;
    }
    return true;
}

bool createDescriptors() {
    const auto device = graphics::internal::context.device;
    const VkShaderStageFlags both_stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    const VkDescriptorSetLayoutBinding scene_bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, both_stages, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
    };
    const VkDescriptorSetLayoutBinding model_bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, both_stages, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
    };
    const VkDescriptorSetLayoutCreateInfo scene_layout = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3, .pBindings = scene_bindings,
    };
    const VkDescriptorSetLayoutCreateInfo model_layout = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2, .pBindings = model_bindings,
    };
    if (!check(vkCreateDescriptorSetLayout(device, &scene_layout, nullptr, &descriptor_layouts[0]),
               "vkCreateDescriptorSetLayout(scene)") ||
        !check(vkCreateDescriptorSetLayout(device, &model_layout, nullptr, &descriptor_layouts[1]),
               "vkCreateDescriptorSetLayout(model)")) return false;

    const VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, material_count},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, material_count},
    };
    const VkDescriptorPoolCreateInfo pool = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = material_count + 1, .poolSizeCount = 4, .pPoolSizes = pool_sizes,
    };
    if (!check(vkCreateDescriptorPool(device, &pool, nullptr, &descriptor_pool),
               "vkCreateDescriptorPool")) return false;
    std::array<VkDescriptorSetLayout, material_count + 1> layouts;
    layouts.fill(descriptor_layouts[1]);
    layouts[0] = descriptor_layouts[0];
    const VkDescriptorSetAllocateInfo allocate = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool, .descriptorSetCount = uint32_t(layouts.size()),
        .pSetLayouts = layouts.data(),
    };
    std::array<VkDescriptorSet, material_count + 1> sets{};
    if (!check(vkAllocateDescriptorSets(device, &allocate, sets.data()),
               "vkAllocateDescriptorSets")) return false;
    scene_descriptor = sets[0];
    if (!createBuffer(sizeof(SceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, scene_buffer) ||
        !createBuffer(model_stride * object_count, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, model_buffer) ||
        !createBuffer(sizeof(PointData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, point_buffer) ||
        !createBuffer(sizeof(SpotData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, spot_buffer)) return false;
    bindBuffer(scene_descriptor, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, scene_buffer);
    bindBuffer(scene_descriptor, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, point_buffer);
    bindBuffer(scene_descriptor, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, spot_buffer);
    for (size_t i = 0; i < materials.size(); ++i) {
        auto& material = materials[i];
        material.descriptor = sets[i + 1];
        bindBuffer(material.descriptor, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                    model_buffer, sizeof(ModelData));
        const VkDescriptorImageInfo image = {texture_sampler, material.albedo.view,
                                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        const VkWriteDescriptorSet write = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = material.descriptor, .dstBinding = 1, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &image,
        };
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }
    return true;
}

// Загрузка шейдеров и настройка графического конвейера
VkShaderModule loadShader(const char* filename) {
    const auto path = std::filesystem::path("shaders") / filename;
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("Cannot open shader: " + path.string());
    const std::streamsize size = stream.tellg();
    if (size <= 0 || size % 4 != 0) throw std::runtime_error("Invalid SPIR-V shader");
    std::vector<uint32_t> code(size_t(size) / sizeof(uint32_t));
    stream.seekg(0);
    if (!stream.read(reinterpret_cast<char*>(code.data()), size))
        throw std::runtime_error("Cannot read shader: " + path.string());
    const VkShaderModuleCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size_t(size), .pCode = code.data(),
    };
    VkShaderModule module = VK_NULL_HANDLE;
    if (!check(vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &module),
               "vkCreateShaderModule")) throw std::runtime_error("Cannot create shader module");
    return module;
}

bool createPipeline(bool markers, VkPrimitiveTopology topology, VkPipeline& output) {
    const auto& context = graphics::internal::context;

    VkShaderModule vertex = VK_NULL_HANDLE, fragment = VK_NULL_HANDLE;
    try {
        vertex = loadShader(markers ? "marker.vert.spv" : "cylinder.vert.spv");
        fragment = loadShader(markers ? "marker.frag.spv" : "cylinder.frag.spv");
    } catch (...) {
        if (vertex) vkDestroyShaderModule(context.device, vertex, nullptr);
        throw;
    }
    const VkPipelineShaderStageCreateInfo stages[] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vertex, .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fragment, .pName = "main"},
    };

    const VkVertexInputBindingDescription binding = {
        0, uint32_t(markers ? sizeof(lighting::MarkerVertex) : sizeof(Vertex)), VK_VERTEX_INPUT_RATE_VERTEX,
    };
    const VkVertexInputAttributeDescription attributes[] = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, uint32_t(offsetof(Vertex, position))},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, uint32_t(offsetof(Vertex, color))},
        {2, 0, VK_FORMAT_R32G32B32_SFLOAT, uint32_t(offsetof(Vertex, normal))},
        {3, 0, VK_FORMAT_R32G32_SFLOAT, uint32_t(offsetof(Vertex, uv))},
    };
    const VkPipelineVertexInputStateCreateInfo input = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = markers ? 2u : 4u, .pVertexAttributeDescriptions = attributes,
    };

    const VkPipelineInputAssemblyStateCreateInfo assembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = topology,
    };
    const VkPipelineViewportStateCreateInfo viewport = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1, .scissorCount = 1,
    };
    const VkPipelineRasterizationStateCreateInfo raster = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VkCullModeFlags(markers ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT),
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f,
    };

    const VkPipelineMultisampleStateCreateInfo samples = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    const VkPipelineDepthStencilStateCreateInfo depth = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE, .depthCompareOp = VK_COMPARE_OP_LESS,
    };

    const VkPipelineColorBlendAttachmentState attachment = {
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    const VkPipelineColorBlendStateCreateInfo blend = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1, .pAttachments = &attachment,
    };

    const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    const VkPipelineDynamicStateCreateInfo dynamic = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2, .pDynamicStates = dynamic_states,
    };

    const VkGraphicsPipelineCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2, .pStages = stages,
        .pVertexInputState = &input, .pInputAssemblyState = &assembly,
        .pViewportState = &viewport, .pRasterizationState = &raster,
        .pMultisampleState = &samples, .pDepthStencilState = &depth,
        .pColorBlendState = &blend, .pDynamicState = &dynamic,
        .layout = pipeline_layout, .renderPass = context.render_pass,
    };
    const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
                                                     &info, nullptr, &output);
    vkDestroyShaderModule(context.device, fragment, nullptr);
    vkDestroyShaderModule(context.device, vertex, nullptr);
    return check(result, "vkCreateGraphicsPipelines");
}

// Анимация и матрицы преобразования
void stepAnimation(double delta) {
    if (animate && playing) animation_time += delta * animation_speed;
}

glm::mat4 modelMatrix(uint32_t index) {
    const auto& object = objects[index];
    glm::vec3 position = object.position, rotation = object.rotation;
    if (animate && index < cylinder_count) {
        const float time = float(animation_time);
        const float phase = time + float(index) * std::numbers::pi_v<float>;
        // Восьмёрка в плоскости XZ с отдельным колебанием по высоте
        position += glm::vec3(path_radius * std::sin(phase),
                              path_height * std::sin(3.0f * phase),
                              0.5f * path_radius * std::sin(2.0f * phase));
        rotation.y += spin_speed * time;
        rotation.x += 15.0f * std::sin(phase);
    }

    glm::mat4 model = glm::translate(glm::mat4(1.0f), position);
    model = glm::rotate(model, glm::radians(rotation.x), {1.0f, 0.0f, 0.0f});
    model = glm::rotate(model, glm::radians(rotation.y), {0.0f, 1.0f, 0.0f});
    model = glm::rotate(model, glm::radians(rotation.z), {0.0f, 0.0f, 1.0f});
    return glm::scale(model, object.scale);
}

void updateUniforms(float aspect) {
    const SceneData scene_data = {
        projectionMatrix(camera, aspect, projection_type, ortho_height) * viewMatrix(camera),
        glm::vec4(camera.position, 1.0f), glm::vec4(forward(camera), float(projection_type)),
        glm::vec4(lights.ambient_color * lights.ambient_intensity, 0.0f),
        glm::vec4(lighting::lightDirection(lights.directional_direction), 0.0f),
        glm::vec4(lights.directional_color * lights.directional_intensity, 0.0f),
        {int(lights.point_lights.size()), int(lights.spot_lights.size()), 0, 0},
    };
    writeBuffer(scene_buffer, &scene_data, sizeof(scene_data));
    for (uint32_t i = 0; i < object_count; ++i) {
        const auto model = modelMatrix(i);
        const ModelData data = {model, glm::transpose(glm::inverse(model)),
            glm::vec4(objects[i].color, objects[i].procedural_color ? 1.0f : 0.0f),
            glm::vec4(objects[i].specular, objects[i].shininess)};
        writeBuffer(model_buffer, &data, sizeof(data), i * model_stride);
    }

    std::vector<PointData> points;
    for (const auto& light : lights.point_lights)
        points.push_back({glm::vec4(light.position, 1.0f), glm::vec4(light.color, light.intensity)});
    std::vector<SpotData> spots;
    for (const auto& light : lights.spot_lights)
        spots.push_back({glm::vec4(light.position, 1.0f),
                         glm::vec4(lighting::lightDirection(light.direction), 0.0f),
                         glm::vec4(light.color, light.intensity),
                         {std::cos(glm::radians(light.inner_angle)),
                          std::cos(glm::radians(light.outer_angle)), 0.0f, 0.0f}});
    const VkDeviceSize point_bytes = std::max(size_t(1), points.size()) * sizeof(PointData);
    const VkDeviceSize spot_bytes = std::max(size_t(1), spots.size()) * sizeof(SpotData);
    if (point_bytes > max_storage_range || spot_bytes > max_storage_range)
        throw std::runtime_error("Light data exceeds device maxStorageBufferRange");
    ensureBuffer(point_buffer, point_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ensureBuffer(spot_buffer, spot_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    bindBuffer(scene_descriptor, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, point_buffer);
    bindBuffer(scene_descriptor, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, spot_buffer);
    if (!points.empty()) writeBuffer(point_buffer, points.data(), points.size() * sizeof(PointData));
    if (!spots.empty()) writeBuffer(spot_buffer, spots.data(), spots.size() * sizeof(SpotData));

    marker_triangle_count = marker_line_count = 0;
    marker_bytes = 0;
    if (lights.show_markers) {
        const auto markers = lighting::makeLightMarkers(lights.point_lights, lights.spot_lights);
        marker_triangle_count = markers.triangle_vertices;
        marker_line_count = uint32_t(markers.vertices.size()) - marker_triangle_count;
        if (!markers.vertices.empty()) {
            marker_bytes = markers.vertices.size() * sizeof(lighting::MarkerVertex);
            ensureBuffer(marker_staging, marker_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            ensureBuffer(marker_buffer, marker_bytes,
                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
            writeBuffer(marker_staging, markers.vertices.data(), size_t(marker_bytes));
        }
    }
}

}

// Создание и освобождение ресурсов
bool initialize() {
    resetScene();
    ImGui::StyleColorsDark();
    ImGui::GetStyle().FrameRounding = 3.0f;
    ImGui::GetStyle().WindowRounding = 5.0f;

    auto mesh = makeCylinder();
    index_count = uint32_t(mesh.indices.size());
    const auto floor = makeFloor();
    const uint32_t floor_vertex_offset = uint32_t(mesh.vertices.size());
    mesh.vertices.insert(mesh.vertices.end(), floor.vertices.begin(), floor.vertices.end());
    for (uint32_t index : floor.indices) mesh.indices.push_back(floor_vertex_offset + index);
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(graphics::internal::context.physical_device, &properties);
    max_storage_range = properties.limits.maxStorageBufferRange;
    const VkDeviceSize alignment = properties.limits.minUniformBufferOffsetAlignment;
    model_stride = (sizeof(ModelData) + alignment - 1) / alignment * alignment;
    try {
        if (model_stride * (object_count - 1) > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("Dynamic uniform offset exceeds uint32_t");
        if (!loadMaterials() || !createDescriptors()) { shutdown(); return false; }
        const VkPushConstantRange sampling_range = {
            VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SamplingParameters),
        };
        const VkPipelineLayoutCreateInfo layout = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = uint32_t(descriptor_layouts.size()), .pSetLayouts = descriptor_layouts.data(),
            .pushConstantRangeCount = 1, .pPushConstantRanges = &sampling_range,
        };
        if (!check(vkCreatePipelineLayout(graphics::internal::context.device, &layout, nullptr, &pipeline_layout),
                   "vkCreatePipelineLayout")) { shutdown(); return false; }
        if (!resources::uploadBuffer(mesh.vertices.data(), mesh.vertices.size() * sizeof(Vertex),
                                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer) ||
            !resources::uploadBuffer(mesh.indices.data(), mesh.indices.size() * sizeof(uint32_t),
                                      VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer) ||
            !createPipeline(false, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, pipeline) ||
            !createPipeline(true, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, marker_pipeline) ||
            !createPipeline(true, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, line_pipeline)) {
            shutdown();
            return false;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        shutdown();
        return false;
    }
    std::cout << "Lab 3, variant 8: " << mesh.vertices.size() << " vertices, "
              << mesh.indices.size() / 3 << " triangles, " << material_count
              << " materials, dynamic UBO stride " << model_stride << " bytes\n";
    return true;
}

void shutdown() {
    const auto& context = graphics::internal::context;
    vkQueueWaitIdle(context.graphics_queue);
    if (marker_pipeline) vkDestroyPipeline(context.device, marker_pipeline, nullptr);
    if (line_pipeline) vkDestroyPipeline(context.device, line_pipeline, nullptr);
    if (pipeline) vkDestroyPipeline(context.device, pipeline, nullptr);
    if (pipeline_layout) vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
    if (descriptor_pool) vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
    for (auto layout : descriptor_layouts)
        if (layout) vkDestroyDescriptorSetLayout(context.device, layout, nullptr);
    pipeline = VK_NULL_HANDLE;
    pipeline_layout = VK_NULL_HANDLE;
    descriptor_pool = VK_NULL_HANDLE;
    descriptor_layouts = {};
    scene_descriptor = VK_NULL_HANDLE;
    marker_pipeline = line_pipeline = VK_NULL_HANDLE;

    for (auto& material : materials) {
        resources::destroyTexture(material.albedo);
        material.descriptor = VK_NULL_HANDLE;
    }
    if (texture_sampler) vkDestroySampler(context.device, texture_sampler, nullptr);
    texture_sampler = VK_NULL_HANDLE;
    destroyBuffer(model_buffer);
    destroyBuffer(scene_buffer);
    destroyBuffer(point_buffer);
    destroyBuffer(spot_buffer);
    destroyBuffer(marker_buffer);
    destroyBuffer(marker_staging);
    marker_bytes = 0;
    destroyBuffer(index_buffer);
    destroyBuffer(vertex_buffer);
}

// Интерфейс и обработка кадра
void update(double time) {
    const double delta = previous_time < 0.0 ? 0.0 : time - previous_time;
    previous_time = time;
    const auto& io = ImGui::GetIO();
    constexpr auto clamp = ImGuiSliderFlags_AlwaysClamp;
    ImGui::SetNextWindowPos({12.0f, 12.0f}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({panel_width, std::max(100.0f, io.DisplaySize.y - 24.0f)}, ImGuiCond_Always);
    ImGui::Begin("Lab 3 | Texture mapping", nullptr,
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
    ImGui::PushItemWidth(175.0f);
    ImGui::TextUnformatted("Cylinder: 50 points on each rim");
    ImGui::TextUnformatted("204 vertices / 200 triangles");
    ImGui::TextUnformatted("Floor: 16 x 16, full image");
    if (ImGui::Button("Reset scene", {-1.0f, 0.0f})) resetScene();

    if (ImGui::CollapsingHeader("Projection and camera")) {
        ImGui::Combo("Projection", &projection_type, "Perspective\0Orthographic\0");
        ImGui::SliderFloat3("Camera position", &camera.position.x, -20.0f, 20.0f, "%.2f", clamp);
        ImGui::SliderFloat("Yaw", &camera.yaw, -180.0f, 180.0f, "%.0f deg", clamp);
        ImGui::SliderFloat("Pitch", &camera.pitch, -89.0f, 89.0f, "%.0f deg", clamp);
        if (projection_type == 0)
            ImGui::SliderFloat("FOV", &camera.fov, 20.0f, 100.0f, "%.0f deg", clamp);
        else ImGui::SliderFloat("View height", &ortho_height, 2.0f, 16.0f, "%.2f", clamp);
    }

    if (ImGui::CollapsingHeader("Object and material", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Combo("Selected", &selected_object, "Cylinder 1\0Cylinder 2\0Floor\0");
        auto& object = objects[size_t(selected_object)];
        ImGui::Checkbox("Visible", &object.visible);
        std::array<const char*, material_count> material_names;
        for (size_t i = 0; i < material_files.size(); ++i) material_names[i] = material_files[i].name;
        ImGui::Combo("Texture", &object.material, material_names.data(), int(material_count));
        ImGui::Checkbox("Animated sampling", &object.sampling_effect);
        ImGui::BeginDisabled(!object.sampling_effect);
        ImGui::SliderFloat("Effect strength", &object.sampling_strength, 0.0f, 1.0f, "%.2f", clamp);
        ImGui::SliderFloat("Effect speed", &object.sampling_speed, 0.0f, 3.0f, "%.2fx", clamp);
        ImGui::EndDisabled();
        ImGui::SliderFloat3("Position", &object.position.x, -4.0f, 4.0f, "%.2f", clamp);
        ImGui::SliderFloat3("Rotation", &object.rotation.x, -180.0f, 180.0f, "%.0f", clamp);
        ImGui::SliderFloat3("Scale", &object.scale.x, 0.1f, 3.0f, "%.2f", clamp);
        ImGui::ColorEdit3("Albedo tint", &object.color.x);
        ImGui::ColorEdit3("Specular", &object.specular.x);
        ImGui::SliderFloat("Shininess", &object.shininess, 1.0f, 256.0f, "%.0f", clamp);
        ImGui::Checkbox("Procedural vertex colors", &object.procedural_color);
    }

    lighting::drawInterface(lights, max_storage_range);

    if (ImGui::CollapsingHeader("Animation")) {
        ImGui::Checkbox("Animate both cylinders", &animate);
        ImGui::BeginDisabled(!animate);
        if (ImGui::Button(playing ? "Pause" : "Play")) playing = !playing;
        ImGui::SameLine();
        if (ImGui::Button("Restart")) animation_time = 0.0;
        ImGui::SliderFloat("Speed", &animation_speed, 0.1f, 3.0f, "%.2fx", clamp);
        ImGui::SliderFloat("Path radius", &path_radius, 0.0f, 2.0f, "%.2f", clamp);
        ImGui::SliderFloat("Path height", &path_height, 0.0f, 1.5f, "%.2f", clamp);
        ImGui::SliderFloat("Spin", &spin_speed, 0.0f, 180.0f, "%.0f deg/s", clamp);
        ImGui::EndDisabled();
        ImGui::Text("Animation time: %.2f s", animation_time);
    }
    ImGui::Separator();
    ImGui::Text("%.1f FPS", io.Framerate);
    ImGui::PopItemWidth();
    ImGui::End();
    sanitizeScene();
    stepAnimation(delta);
    sampling_time += std::max(0.0, delta);
}

void render(const graphics::internal::FrameData& frame) {
    const auto& context = graphics::internal::context;
    const auto cmd = frame.command_buffer;
    const auto extent = context.swapchain_extent;
    const float left = extent.width > 700 ? scene_left : 0.0f;
    const float width = float(extent.width) - left, height = float(extent.height);

    // prepare() уже дождался завершения предыдущего кадра.
    updateUniforms(width / height);

    if (!check(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer"))
        throw std::runtime_error("Cannot reset command buffer");
    const VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (!check(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer"))
        throw std::runtime_error("Cannot begin command buffer");

    if (marker_bytes > 0) {
        const VkBufferCopy copy = {0, 0, marker_bytes};
        vkCmdCopyBuffer(cmd, marker_staging.handle, marker_buffer.handle, 1, &copy);
        const VkBufferMemoryBarrier barrier = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = marker_buffer.handle, .offset = 0, .size = marker_bytes,
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                             0, 0, nullptr, 1, &barrier, 0, nullptr);
    }

    VkClearValue clear[2]{};
    clear[0].color = {{0.035f, 0.045f, 0.07f, 1.0f}};
    clear[1].depthStencil = {1.0f, 0};
    const VkRenderPassBeginInfo pass = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = context.render_pass, .framebuffer = frame.framebuffer,
        .renderArea = {.extent = extent}, .clearValueCount = 2, .pClearValues = clear,
    };
    vkCmdBeginRenderPass(cmd, &pass, VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport = {left, 0.0f, width, height, 0.0f, 1.0f};
    const VkRect2D scissor = {{int32_t(left), 0}, {uint32_t(width), extent.height}};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline_layout, 0, 1, &scene_descriptor, 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.handle, &offset);
    vkCmdBindIndexBuffer(cmd, index_buffer.handle, 0, VK_INDEX_TYPE_UINT32);
    for (uint32_t i = 0; i < object_count; ++i) {
        if (!objects[i].visible) continue;
        const auto& object = objects[i];
        const VkDescriptorSet descriptor = materials[size_t(object.material)].descriptor;
        const uint32_t dynamic_offset = uint32_t(i * model_stride);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline_layout, 1, 1, &descriptor, 1, &dynamic_offset);
        const SamplingParameters sampling = {float(sampling_time), object.sampling_strength,
                                              object.sampling_speed, object.sampling_effect ? 1u : 0u};
        vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(sampling), &sampling);
        vkCmdDrawIndexed(cmd, i == floor_object ? 6u : index_count, 1,
                         i == floor_object ? index_count : 0u, 0, 0);
    }

    if (marker_triangle_count + marker_line_count > 0) {
        vkCmdBindVertexBuffers(cmd, 0, 1, &marker_buffer.handle, &offset);
        if (marker_triangle_count > 0) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, marker_pipeline);
            vkCmdDraw(cmd, marker_triangle_count, 1, 0, 0);
        }
        if (marker_line_count > 0) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, line_pipeline);
            vkCmdDraw(cmd, marker_line_count, 1, marker_triangle_count, 0);
        }
    }

    vkCmdEndRenderPass(cmd);
    if (!check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer"))
        throw std::runtime_error("Cannot end command buffer");
}

}
