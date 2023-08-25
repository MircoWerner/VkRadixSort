#pragma once

#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "GPUContext.h"
#include <vulkan/vulkan_core.h>

namespace engine {
    class Buffer {
    public:
        struct BufferSettings {
            uint32_t m_sizeBytes;
            VkBufferUsageFlags m_bufferUsages;
            VkMemoryPropertyFlags m_memoryProperties;
            std::optional<VkMemoryAllocateFlagBits> m_memoryAllocateFlagBits{};

            std::string m_name = "undefined";
        };

        Buffer(GPUContext *gpuContext, BufferSettings settings) : m_gpuContext(gpuContext), m_bufferSettings(std::move(settings)) {
            createBuffer();

//            m_gpuContext->getDebug()->setName(m_buffer, m_bufferSettings.m_name);
//            m_gpuContext->getDebug()->setName(m_bufferMemory, m_bufferSettings.m_name);
        }

        ~Buffer() {
            release();
        }

        void release() {
            if (m_buffer) {
                vkDestroyBuffer(m_gpuContext->m_device, m_buffer, nullptr);
            }
            if (m_bufferMemory) {
                vkFreeMemory(m_gpuContext->m_device, m_bufferMemory, nullptr);
            }
            m_buffer = nullptr;
            m_bufferMemory = nullptr;
        }

        static std::shared_ptr<Buffer> fillDeviceWithStagingBuffer(GPUContext *gpuContext, const BufferSettings& settings, void *data) { // upload
            Buffer stagingBuffer(gpuContext, {settings.m_sizeBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT});

            void *stagingMemory;
            vkMapMemory(gpuContext->m_device, stagingBuffer.m_bufferMemory, 0, settings.m_sizeBytes, 0, &stagingMemory); // memory-mapped I/O
            memcpy(stagingMemory, data, settings.m_sizeBytes);
            vkUnmapMemory(gpuContext->m_device, stagingBuffer.m_bufferMemory);

            auto buffer = std::make_shared<Buffer>(gpuContext, settings);

            copyBuffer(gpuContext, stagingBuffer.m_buffer, buffer->m_buffer, settings.m_sizeBytes); // copy contents from staging buffer to high performance memory on GPU, which cannot be accessed directly by the CPU (therefore the staging buffer)

            stagingBuffer.release();

            return buffer;
        }

        void downloadWithStagingBuffer(void *data) {
            Buffer stagingBuffer(m_gpuContext, {m_bufferSettings.m_sizeBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT});

            copyBuffer(m_gpuContext, m_buffer, stagingBuffer.m_buffer, m_bufferSettings.m_sizeBytes); // copy contents from staging buffer to high performance memory on GPU, which cannot be accessed directly by the CPU (therefore the staging buffer)

            stagingBuffer.download(data);

            stagingBuffer.release();
        }

        void download(void *data) {
            void *stagingMemory;
            vkMapMemory(m_gpuContext->m_device, m_bufferMemory, 0, m_bufferSettings.m_sizeBytes, 0, &stagingMemory); // memory-mapped I/O
            memcpy(data, stagingMemory, m_bufferSettings.m_sizeBytes);
            vkUnmapMemory(m_gpuContext->m_device, m_bufferMemory);
        }

        void updateHostMemory(uint32_t sizeBytes, void *data) {
            void *memory;
            vkMapMemory(m_gpuContext->m_device, m_bufferMemory, 0, sizeBytes, 0, &memory); // memory-mapped I/O
            memcpy(memory, data, sizeBytes);
            vkUnmapMemory(m_gpuContext->m_device, m_bufferMemory);
        }

        void *mapHostMemory() {
            void *memory;
            vkMapMemory(m_gpuContext->m_device, m_bufferMemory, 0, m_bufferSettings.m_sizeBytes, 0, &memory); // memory-mapped I/O
            return memory;
        }

        void unmapHostMemory() {
            vkUnmapMemory(m_gpuContext->m_device, m_bufferMemory);
        }


        VkBuffer getBuffer() {
            return m_buffer;
        }

        [[nodiscard]] uint32_t getSizeBytes() const {
            return m_bufferSettings.m_sizeBytes;
        }

        VkDeviceAddress getDeviceAddress() {
            VkBufferDeviceAddressInfo addressInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = m_buffer};
            return vkGetBufferDeviceAddress(m_gpuContext->m_device, &addressInfo);
        }

    private:
        GPUContext *m_gpuContext;

        VkBuffer m_buffer = nullptr;
        VkDeviceMemory m_bufferMemory = nullptr;

        BufferSettings m_bufferSettings;

        void createBuffer() {
            VkBufferCreateInfo bufferInfo{};
            bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferInfo.size = m_bufferSettings.m_sizeBytes;
            bufferInfo.usage = m_bufferSettings.m_bufferUsages;
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            if (vkCreateBuffer(m_gpuContext->m_device, &bufferInfo, nullptr, &m_buffer) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create buffer!");
            }

            VkMemoryRequirements memRequirements;
            vkGetBufferMemoryRequirements(m_gpuContext->m_device, m_buffer, &memRequirements);

            VkMemoryAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = findMemoryType(m_gpuContext->m_physicalDevice, memRequirements.memoryTypeBits, m_bufferSettings.m_memoryProperties);
            VkMemoryAllocateFlagsInfo *pMemoryAllocateFlagsInfo = nullptr;
            VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo{};
            if (m_bufferSettings.m_memoryAllocateFlagBits.has_value()) {
                memoryAllocateFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
                memoryAllocateFlagsInfo.flags = m_bufferSettings.m_memoryAllocateFlagBits.value();
                pMemoryAllocateFlagsInfo = &memoryAllocateFlagsInfo;
            }
            allocInfo.pNext = pMemoryAllocateFlagsInfo;

            if (vkAllocateMemory(m_gpuContext->m_device, &allocInfo, nullptr, &m_bufferMemory) != VK_SUCCESS) {
                throw std::runtime_error("Failed to allocate buffer memory!");
            }

            vkBindBufferMemory(m_gpuContext->m_device, m_buffer, m_bufferMemory, 0);
        }

        static uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
            VkPhysicalDeviceMemoryProperties memProperties;
            vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

            for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
                if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                    return i;
                }
            }

            throw std::runtime_error("Failed to find suitable memory type!");
        }

        static void copyBuffer(GPUContext *gpuContext, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = 0; // optional
            copyRegion.dstOffset = 0; // optional
            copyRegion.size = size;

            gpuContext->executeCommands([&](VkCommandBuffer commandBuffer) {
                vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
            });
        }
    };
} // namespace raven