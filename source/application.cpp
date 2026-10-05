#include "application.hpp"

#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace application {
namespace {

// Геометрия и размеры интерфейса
constexpr uint32_t segments = 50;
constexpr uint32_t object_count = 2;
constexpr float panel_width = 310.0f;
constexpr float scene_left = 334.0f;

// Начальные настройки камеры
constexpr float default_camera_dist = 9.0f;
constexpr float default_camera_yaw = 25.0f;
constexpr float default_camera_pitch = 18.0f;
constexpr float default_fov = 45.0f;
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
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    void* mapped = nullptr;
};

// Совпадает с блоком std140 в шейдере: матрица 64 байта и цвет 16 байт
struct alignas(16) UniformData {
    glm::mat4 mvp;
    glm::vec4 tint;
};
static_assert(sizeof(UniformData) == 80);

struct Object {
    glm::vec3 position{0.0f};
    glm::vec3 rotation{0.0f}; // Углы в градусах
    glm::vec3 scale{1.0f};
    glm::vec3 color{1.0f};
    bool visible = true;
    bool procedural_color = true;
};

struct ObjectGpu {
    Buffer uniform;
    VkDescriptorSet descriptor = VK_NULL_HANDLE;
};

// Ресурсы Vulkan
Buffer vertex_buffer, index_buffer;
uint32_t index_count = 0;
VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;
std::array<ObjectGpu, object_count> gpu_objects;

// Параметры сцены и камеры
std::array<Object, object_count> objects;
int selected_object = 0;
int projection_type = 0;
float camera_distance = default_camera_dist;
float camera_yaw = default_camera_yaw;
float camera_pitch = default_camera_pitch;
float fov = default_fov;
float ortho_height = default_ortho_height;

// Состояние и параметры анимации
bool animate = false;
bool playing = true;
float animation_speed = default_anim_speed;
float path_radius = default_path_radius;
float path_height = default_path_height;
float spin_speed = default_spin_speed;
double animation_time = 0.0;
double previous_time = -1.0;

bool check(VkResult result, const char* operation) {
    if (result == VK_SUCCESS) return true;
    std::cerr << operation << " failed (VkResult " << result << ")\n";
    return false;
}

void resetScene() {
    objects = {};
    objects[0].position = {-1.5f, 0.0f, 0.0f};
    objects[0].rotation = {12.0f, 0.0f, -10.0f};
    objects[1].position = {1.5f, 0.0f, 0.0f};
    objects[1].rotation = {-15.0f, 20.0f, 15.0f};
    objects[1].scale = glm::vec3(0.8f);
    objects[1].color = {0.5f, 0.85f, 1.0f};

    selected_object = 0;
    projection_type = 0;
    camera_distance = default_camera_dist;
    camera_yaw = default_camera_yaw;
    camera_pitch = default_camera_pitch;
    fov = default_fov;
    ortho_height = default_ortho_height;

    animate = false;
    playing = true;
    animation_speed = default_anim_speed;
    path_radius = default_path_radius;
    path_height = default_path_height;
    spin_speed = default_spin_speed;
    animation_time = 0.0;
    previous_time = -1.0;
}

// Цилиндр вдоль Y: радиус 1, высота 2, по 50 точек на каждом основании
Mesh makeCylinder() {
    Mesh mesh;
    mesh.vertices.reserve(2 * segments + 2);
    mesh.indices.reserve(12 * segments);

    auto addVertex = [&](glm::vec3 pos) {
        // Преобразуем локальные координаты [-1, 1] в компоненты RGB [0, 1]
        mesh.vertices.push_back({pos, (pos + glm::vec3(1.0f)) * 0.5f});
    };

    // Добавляем вершины парами: нижняя, затем верхняя
    for (uint32_t i = 0; i < segments; ++i) {
        const float angle = 2.0f * std::numbers::pi_v<float> * float(i) / float(segments);
        const float x = std::cos(angle), z = std::sin(angle);
        addVertex({x, -1.0f, z});
        addVertex({x,  1.0f, z});
    }

    const uint32_t bottom_center = uint32_t(mesh.vertices.size());
    addVertex({0.0f, -1.0f, 0.0f});
    const uint32_t top_center = uint32_t(mesh.vertices.size());
    addVertex({0.0f, 1.0f, 0.0f});

    for (uint32_t i = 0; i < segments; ++i) {
        const uint32_t next = (i + 1) % segments;
        const uint32_t bottom = 2 * i, top = bottom + 1;
        const uint32_t next_bottom = 2 * next, next_top = next_bottom + 1;
        // Два треугольника боковой стенки и по одному на крышках; обход наружу
        mesh.indices.insert(mesh.indices.end(), {
            bottom, top, next_bottom,
            top, next_top, next_bottom,
            bottom_center, bottom, next_bottom,
            top_center, next_top, top,
        });
    }
    return mesh;
}

// Создание буферов и привязка данных к шейдерам
bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& buffer) {
    const VkBufferCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size, .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    const VmaAllocationCreateInfo allocation_info = {
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
    };
    VmaAllocationInfo allocation{};
    if (!check(vmaCreateBuffer(graphics::internal::context.allocator, &info,
                              &allocation_info, &buffer.handle, &buffer.allocation,
                              &allocation), "vmaCreateBuffer")) return false;
    buffer.mapped = allocation.pMappedData;
    return true;
}

void writeBuffer(const Buffer& buffer, const void* data, size_t size) {
    std::memcpy(buffer.mapped, data, size);
    // Передаём записи cpu в gpu, даже если память не является когерентной
    if (!check(vmaFlushAllocation(graphics::internal::context.allocator,
                                 buffer.allocation, 0, size), "vmaFlushAllocation"))
        throw std::runtime_error("Could not flush GPU buffer");
}

bool createDescriptors() {
    // Каждый цилиндр получает свой набор дескрипторов и буфер матрицы и цвета
    const auto device = graphics::internal::context.device;

    // Описываем набор: один uniform-буфер для вершинного шейдера
    const VkDescriptorSetLayoutBinding binding = {
        .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
    };
    const VkDescriptorSetLayoutCreateInfo layout = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1, .pBindings = &binding,
    };
    if (!check(vkCreateDescriptorSetLayout(device, &layout, nullptr, &descriptor_layout),
               "vkCreateDescriptorSetLayout")) return false;

    // Выделяем по одному набору дескрипторов на объект
    const VkDescriptorPoolSize pool_size = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, object_count};
    const VkDescriptorPoolCreateInfo pool = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = object_count, .poolSizeCount = 1, .pPoolSizes = &pool_size,
    };
    if (!check(vkCreateDescriptorPool(device, &pool, nullptr, &descriptor_pool),
               "vkCreateDescriptorPool")) return false;
    const std::array<VkDescriptorSetLayout, object_count> layouts = {descriptor_layout, descriptor_layout};
    const VkDescriptorSetAllocateInfo allocate = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool, .descriptorSetCount = object_count,
        .pSetLayouts = layouts.data(),
    };
    std::array<VkDescriptorSet, object_count> sets{};
    if (!check(vkAllocateDescriptorSets(device, &allocate, sets.data()),
               "vkAllocateDescriptorSets")) return false;

    // Привязываем каждый набор к буферу своего цилиндра
    for (uint32_t i = 0; i < object_count; ++i) {
        auto& gpu = gpu_objects[i];
        gpu.descriptor = sets[i];
        if (!createBuffer(sizeof(UniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, gpu.uniform)) return false;
        const VkDescriptorBufferInfo buffer_info = {gpu.uniform.handle, 0, sizeof(UniformData)};
        const VkWriteDescriptorSet write = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = gpu.descriptor, .dstBinding = 0, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &buffer_info,
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

bool createPipeline() {
    const auto& context = graphics::internal::context;

    // Указываем, какие наборы ресурсов будут доступны шейдерам
    const VkPipelineLayoutCreateInfo layout = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &descriptor_layout,
    };
    if (!check(vkCreatePipelineLayout(context.device, &layout, nullptr, &pipeline_layout),
               "vkCreatePipelineLayout")) return false;

    // Загружаем вершинный и фрагментный шейдеры
    VkShaderModule vertex = VK_NULL_HANDLE, fragment = VK_NULL_HANDLE;
    try {
        vertex = loadShader("cylinder.vert.spv");
        fragment = loadShader("cylinder.frag.spv");
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

    // Указываем, где в Vertex находятся координаты и цвет
    const VkVertexInputBindingDescription binding = {0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    const VkVertexInputAttributeDescription attributes[] = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, uint32_t(offsetof(Vertex, position))},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, uint32_t(offsetof(Vertex, color))},
    };
    const VkPipelineVertexInputStateCreateInfo input = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = 2, .pVertexAttributeDescriptions = attributes,
    };

    // Собираем треугольники и отсекаем задние грани
    const VkPipelineInputAssemblyStateCreateInfo assembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };
    const VkPipelineViewportStateCreateInfo viewport = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1, .scissorCount = 1,
    };
    const VkPipelineRasterizationStateCreateInfo raster = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f,
    };

    const VkPipelineMultisampleStateCreateInfo samples = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    // Проверка глубины определяет, какая поверхность видна спереди
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

    // Размеры области рисования задаются в render() при каждом кадре
    const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    const VkPipelineDynamicStateCreateInfo dynamic = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2, .pDynamicStates = dynamic_states,
    };

    // Создаём конвейер и освобождаем временные модули шейдеров
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
                                                     &info, nullptr, &pipeline);
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
    if (animate) {
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

glm::mat4 projectionMatrix(float aspect) {
    glm::mat4 projection;
    if (projection_type == 0) {
        projection = glm::perspectiveRH_ZO(glm::radians(fov), aspect, 0.1f, 100.0f);
    } else {
        const float half_height = ortho_height * 0.5f;
        projection = glm::orthoRH_ZO(-half_height * aspect, half_height * aspect,
                                    -half_height, half_height, 0.1f, 100.0f);
    }
    projection[1][1] *= -1.0f;
    return projection;
}

void updateUniforms(float aspect) {
    const float yaw = glm::radians(camera_yaw), pitch = glm::radians(camera_pitch);
    const glm::vec3 cam_pos = camera_distance * glm::vec3(std::cos(pitch) * std::sin(yaw),
                                                        std::sin(pitch),
                                                        std::cos(pitch) * std::cos(yaw));
    const auto view = glm::lookAt(cam_pos, glm::vec3(0.0f), {0.0f, 1.0f, 0.0f});
    const auto projection = projectionMatrix(aspect);

    for (uint32_t i = 0; i < object_count; ++i) {
        const UniformData data = {projection * view * modelMatrix(i),
            glm::vec4(objects[i].color, objects[i].procedural_color ? 1.0f : 0.0f)};
        writeBuffer(gpu_objects[i].uniform, &data, sizeof(data));
    }
}

}

// Создание и освобождение ресурсов
bool initialize() {
    // Геометрию и ресурсы gpu создаём один раз при запуске
    resetScene();
    ImGui::StyleColorsDark();
    ImGui::GetStyle().FrameRounding = 3.0f;
    ImGui::GetStyle().WindowRounding = 5.0f;

    const Mesh mesh = makeCylinder();
    index_count = uint32_t(mesh.indices.size());
    try {
        if (!createBuffer(mesh.vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                          vertex_buffer) ||
            !createBuffer(mesh.indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                          index_buffer) || !createDescriptors() || !createPipeline()) {
            shutdown();
            return false;
        }
        writeBuffer(vertex_buffer, mesh.vertices.data(), mesh.vertices.size() * sizeof(Vertex));
        writeBuffer(index_buffer, mesh.indices.data(), mesh.indices.size() * sizeof(uint32_t));
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        shutdown();
        return false;
    }
    std::cout << "Lab 1, variant 8: " << mesh.vertices.size() << " vertices, "
              << mesh.indices.size() / 3 << " triangles, 2 descriptor sets\n";
    return true;
}

void shutdown() {
    // Сначала ждём завершения GPU, затем освобождаем созданные ресурсы
    const auto& context = graphics::internal::context;
    vkQueueWaitIdle(context.graphics_queue);
    if (pipeline) vkDestroyPipeline(context.device, pipeline, nullptr);
    if (pipeline_layout) vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
    if (descriptor_pool) vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
    if (descriptor_layout) vkDestroyDescriptorSetLayout(context.device, descriptor_layout, nullptr);
    pipeline = VK_NULL_HANDLE;
    pipeline_layout = VK_NULL_HANDLE;
    descriptor_pool = VK_NULL_HANDLE;
    descriptor_layout = VK_NULL_HANDLE;

    auto destroyBuffer = [&](Buffer& buffer) {
        if (buffer.handle) vmaDestroyBuffer(context.allocator, buffer.handle, buffer.allocation);
        buffer = {};
    };
    for (auto& gpu : gpu_objects) {
        destroyBuffer(gpu.uniform);
        gpu.descriptor = VK_NULL_HANDLE;
    }
    destroyBuffer(index_buffer);
    destroyBuffer(vertex_buffer);
}

// Интерфейс и обработка кадра
void update(double time) {
    // Здесь меняем параметры интерфейса; запись в память GPU выполняется в render
    const double delta = previous_time < 0.0 ? 0.0 : time - previous_time;
    previous_time = time;

    ImGui::SetNextWindowPos({12.0f, 12.0f}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({panel_width, ImGui::GetIO().DisplaySize.y - 24.0f}, ImGuiCond_Always);
    ImGui::Begin("Lab 1 | Variant 8", nullptr,
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
    ImGui::PushItemWidth(170.0f);
    ImGui::TextUnformatted("Cylinder: 50 points on each rim");
    ImGui::TextUnformatted("102 vertices / 200 triangles");
    ImGui::TextUnformatted("Two objects / two descriptor sets");
    if (ImGui::Button("Reset scene", {-1.0f, 0.0f})) resetScene();

    if (ImGui::CollapsingHeader("Projection and camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Combo("Projection", &projection_type, "Perspective\0Orthographic\0");
        if (projection_type == 0) ImGui::SliderFloat("FOV", &fov, 20.0f, 90.0f, "%.0f deg");
        else ImGui::SliderFloat("View height", &ortho_height, 2.0f, 12.0f);
        ImGui::SliderFloat("Distance", &camera_distance, 4.0f, 18.0f);
        ImGui::SliderFloat("Camera yaw", &camera_yaw, -180.0f, 180.0f, "%.0f deg");
        ImGui::SliderFloat("Camera pitch", &camera_pitch, -80.0f, 80.0f, "%.0f deg");
    }

    if (ImGui::CollapsingHeader("Object", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Combo("Selected", &selected_object, "Cylinder 1\0Cylinder 2\0");
        auto& object = objects[size_t(selected_object)];
        ImGui::Checkbox("Visible", &object.visible);
        ImGui::SliderFloat3("Position", &object.position.x, -4.0f, 4.0f, "%.2f");
        ImGui::SliderFloat3("Rotation", &object.rotation.x, -180.0f, 180.0f, "%.0f");
        ImGui::SliderFloat3("Scale", &object.scale.x, 0.1f, 3.0f, "%.2f");
        ImGui::ColorEdit3("Color", &object.color.x);
        ImGui::Checkbox("Procedural vertex colors", &object.procedural_color);
        ImGui::TextWrapped("Final RGB = vertex RGB x selected RGB");
    }

    if (ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Animate both cylinders", &animate);
        ImGui::BeginDisabled(!animate);
        if (ImGui::Button(playing ? "Pause" : "Play")) playing = !playing;
        ImGui::SameLine();
        if (ImGui::Button("Restart")) animation_time = 0.0;
        ImGui::SliderFloat("Speed", &animation_speed, 0.1f, 3.0f, "%.2fx");
        ImGui::SliderFloat("Path radius", &path_radius, 0.0f, 2.0f);
        ImGui::SliderFloat("Path height", &path_height, 0.0f, 1.5f);
        ImGui::SliderFloat("Spin", &spin_speed, 0.0f, 180.0f, "%.0f deg/s");
        ImGui::EndDisabled();
        ImGui::TextWrapped("Path: figure-eight with vertical oscillation");
        ImGui::Text("Animation time: %.2f s", animation_time);
    }

    ImGui::Separator();
    ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
    ImGui::PopItemWidth();
    ImGui::End();

    stepAnimation(delta);
}

void render(const graphics::internal::FrameData& frame) {
    const auto& context = graphics::internal::context;
    const auto cmd = frame.command_buffer;
    const auto extent = context.swapchain_extent;
    const float left = extent.width > 700 ? scene_left : 0.0f;
    const float width = float(extent.width) - left, height = float(extent.height);

    // prepare() уже дождался предыдущего кадра: теперь можно обновлять буферы
    // В update() gpu ещё может читать их, поэтому там запись была бы небезопасна
    updateUniforms(width / height);

    // Начинаем запись команд текущего кадра
    if (!check(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer"))
        throw std::runtime_error("Cannot reset command buffer");
    const VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (!check(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer"))
        throw std::runtime_error("Cannot begin command buffer");

    // Очищаем цвет и глубину перед рисованием сцены
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

    // Используем общую геометрию и отдельные параметры каждого цилиндра
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.handle, &offset);
    vkCmdBindIndexBuffer(cmd, index_buffer.handle, 0, VK_INDEX_TYPE_UINT32);
    for (uint32_t i = 0; i < object_count; ++i) {
        if (!objects[i].visible) continue;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline_layout, 0, 1, &gpu_objects[i].descriptor, 0, nullptr);
        vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
    }

    vkCmdEndRenderPass(cmd);
    if (!check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer"))
        throw std::runtime_error("Cannot end command buffer");
}

}
