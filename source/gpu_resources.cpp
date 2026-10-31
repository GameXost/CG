#include "gpu_resources.hpp"

#include <lodepng.h>

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace resources {
namespace {

bool check(VkResult result, const char* operation) {
    if (result == VK_SUCCESS) return true;
    std::cerr << operation << " failed (VkResult " << result << ")\n";
    return false;
}

// Ресурсы загрузки живут до завершения команд копирования.
template <typename Record>
bool submitUpload(Record record) {
    const auto& context = graphics::internal::context;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    const VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = context.graphics_queue_index,
    };
    if (!check(vkCreateCommandPool(context.device, &pool_info, nullptr, &pool),
               "vkCreateCommandPool(upload)")) return false;
    const VkCommandBufferAllocateInfo allocate = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer command = VK_NULL_HANDLE;
    bool success = check(vkAllocateCommandBuffers(context.device, &allocate, &command),
                         "vkAllocateCommandBuffers(upload)");
    const VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (success) success = check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer(upload)");
    if (success) {
        record(command);
        success = check(vkEndCommandBuffer(command), "vkEndCommandBuffer(upload)");
    }
    const VkFenceCreateInfo fence_info = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (success) success = check(vkCreateFence(context.device, &fence_info, nullptr, &fence),
                                 "vkCreateFence(upload)");
    const VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &command,
    };
    if (success) success = check(vkQueueSubmit(context.graphics_queue, 1, &submit, fence),
                                 "vkQueueSubmit(upload)");
    if (success && !check(vkWaitForFences(context.device, 1, &fence, VK_TRUE, UINT64_MAX),
                          "vkWaitForFences(upload)")) {
        vkQueueWaitIdle(context.graphics_queue);
        success = false;
    }
    if (fence) vkDestroyFence(context.device, fence, nullptr);
    vkDestroyCommandPool(context.device, pool, nullptr);
    return success;
}

} // namespace

bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& buffer,
                  bool device_local) {
    if (size == 0 || buffer.handle) return false;
    const VkBufferCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size, .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    const VmaAllocationCreateInfo allocation_info = {
        .flags = device_local ? VmaAllocationCreateFlags(0) :
            VmaAllocationCreateFlags(VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                     VMA_ALLOCATION_CREATE_MAPPED_BIT),
        .usage = device_local ? VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE : VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .requiredFlags = device_local ? VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) : 0u,
    };
    VmaAllocationInfo allocation{};
    if (!check(vmaCreateBuffer(graphics::internal::context.allocator, &info,
                              &allocation_info, &buffer.handle, &buffer.allocation,
                              &allocation), "vmaCreateBuffer")) return false;
    buffer.mapped = allocation.pMappedData;
    buffer.capacity = size;
    return true;
}

void writeBuffer(const Buffer& buffer, const void* data, size_t size, VkDeviceSize offset) {
    if (!buffer.mapped || !data || offset > buffer.capacity || size > buffer.capacity - offset)
        throw std::runtime_error("Invalid mapped buffer write");
    if (size == 0) return;
    std::memcpy(static_cast<unsigned char*>(buffer.mapped) + offset, data, size);
    if (!check(vmaFlushAllocation(graphics::internal::context.allocator,
                                 buffer.allocation, offset, size), "vmaFlushAllocation"))
        throw std::runtime_error("Could not flush GPU buffer");
}

void destroyBuffer(Buffer& buffer) {
    if (buffer.handle) vmaDestroyBuffer(graphics::internal::context.allocator,
                                       buffer.handle, buffer.allocation);
    buffer = {};
}

bool uploadBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage, Buffer& buffer) {
    if (!data || size == 0 || buffer.handle ||
        !(usage & (VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT))) return false;
    Buffer staging;
    try {
        if (!createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging) ||
            !createBuffer(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, buffer, true)) {
            destroyBuffer(staging);
            return false;
        }
        writeBuffer(staging, data, size_t(size));
        const bool success = submitUpload([&](VkCommandBuffer command) {
            const VkBufferCopy copy = {0, 0, size};
            vkCmdCopyBuffer(command, staging.handle, buffer.handle, 1, &copy);
            const VkBufferMemoryBarrier barrier = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VkAccessFlags((usage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT ? VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT : 0) |
                                              (usage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT ? VK_ACCESS_INDEX_READ_BIT : 0)),
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = buffer.handle, .offset = 0, .size = size,
            };
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                                 0, 0, nullptr, 1, &barrier, 0, nullptr);
        });
        destroyBuffer(staging);
        if (!success) destroyBuffer(buffer);
        return success;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        destroyBuffer(staging);
        destroyBuffer(buffer);
        return false;
    }
}

bool loadTexture(const char* filename, VkFormat format, Texture& texture) {
    if (!filename || texture.image ||
        (format != VK_FORMAT_R8G8B8A8_SRGB && format != VK_FORMAT_R8G8B8A8_UNORM)) return false;
    const auto& context = graphics::internal::context;
    Buffer staging;
    try {
        std::vector<unsigned char> pixels;
        unsigned width = 0, height = 0;
        const unsigned error = lodepng::decode(pixels, width, height, filename);
        if (error) {
            std::cerr << "Cannot decode " << filename << ": " << lodepng_error_text(error) << '\n';
            return false;
        }
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(context.physical_device, &properties);
        if (!width || !height || width > properties.limits.maxImageDimension2D ||
            height > properties.limits.maxImageDimension2D ||
            VkDeviceSize(width) * height * 4 != pixels.size()) {
            std::cerr << "Texture dimensions exceed device limits: " << filename << '\n';
            return false;
        }
        VkFormatProperties format_properties{};
        vkGetPhysicalDeviceFormatProperties(context.physical_device, format, &format_properties);
        constexpr VkFormatFeatureFlags features = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        if ((format_properties.optimalTilingFeatures & features) != features) return false;
        if (!createBuffer(pixels.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging)) return false;
        writeBuffer(staging, pixels.data(), pixels.size());
        const VkImageCreateInfo image = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D, .format = format,
            .extent = {width, height, 1}, .mipLevels = 1, .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        const VmaAllocationCreateInfo allocation = {
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        };
        if (!check(vmaCreateImage(context.allocator, &image, &allocation, &texture.image,
                                  &texture.allocation, nullptr), "vmaCreateImage(texture)")) {
            destroyBuffer(staging);
            return false;
        }
        const bool uploaded = submitUpload([&](VkCommandBuffer command) {
            VkImageMemoryBarrier barrier = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = 0, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = texture.image,
                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
            };
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &barrier);
            const VkBufferImageCopy copy = {
                .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                .imageExtent = {width, height, 1},
            };
            vkCmdCopyBufferToImage(command, staging.handle, texture.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &barrier);
        });
        destroyBuffer(staging);
        if (!uploaded) { destroyTexture(texture); return false; }
        const VkImageViewCreateInfo view = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = texture.image, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        };
        if (!check(vkCreateImageView(context.device, &view, nullptr, &texture.view),
                   "vkCreateImageView(texture)")) { destroyTexture(texture); return false; }
        return true;
    } catch (const std::exception& error) {
        std::cerr << filename << ": " << error.what() << '\n';
        destroyBuffer(staging);
        destroyTexture(texture);
        return false;
    }
}

void destroyTexture(Texture& texture) {
    const auto& context = graphics::internal::context;
    if (texture.view) vkDestroyImageView(context.device, texture.view, nullptr);
    if (texture.image) vmaDestroyImage(context.allocator, texture.image, texture.allocation);
    texture = {};
}

} // namespace resources
