#pragma once

#include "graphics_internal.hpp"

#include <cstddef>

namespace resources {

struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize capacity = 0;
};

struct Texture {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& buffer,
                  bool device_local = false);
void writeBuffer(const Buffer& buffer, const void* data, size_t size,
                 VkDeviceSize offset = 0);
void destroyBuffer(Buffer& buffer);
bool uploadBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage,
                  Buffer& buffer);

bool loadTexture(const char* filename, VkFormat format, Texture& texture);
void destroyTexture(Texture& texture);

} // namespace resources
