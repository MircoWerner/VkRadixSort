#pragma once

#include <cstring>
#include <iostream>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan_core.h>

#include "Queues.h"

namespace engine {
    class GPUContext {
    public:
        explicit GPUContext(uint32_t requiredQueueFamilies);

        virtual void init();

        virtual void shutdown();

        VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE; // will be destroyed implicitly when instance is destroyed

        VkDevice m_device{};
        std::shared_ptr<Queues> m_queues;

        uint32_t m_activeIndex = 0;

        [[nodiscard]] uint32_t getMultiBufferedCount() const {
            return MAX_FRAMES_IN_FLIGHT;
        }

        [[nodiscard]] uint32_t getActiveIndex() const {
            return m_activeIndex;
        }

        VkCommandPool m_commandPool{}; // TODO(Mirco): make this a transfer command pool only

        void executeCommands(const std::function<void(VkCommandBuffer)> &recordCommands) {
            VkCommandBufferAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandPool = m_commandPool;
            allocInfo.commandBufferCount = 1;

            VkCommandBuffer commandBuffer;
            vkAllocateCommandBuffers(m_device, &allocInfo, &commandBuffer);

            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            vkBeginCommandBuffer(commandBuffer, &beginInfo);

            recordCommands(commandBuffer);

            vkEndCommandBuffer(commandBuffer);

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;

            vkQueueSubmit(m_queues->getQueue(Queues::TRANSFER), 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(m_queues->getQueue(Queues::TRANSFER));

            vkFreeCommandBuffers(m_device, m_commandPool, 1, &commandBuffer);
        }

    protected:
        VkInstance m_instance{};
        VkDebugUtilsMessengerEXT m_debugMessenger{};

        std::vector<VkCommandBuffer> m_commandBuffers; // destroyed implicitly with the command pool

        virtual std::vector<const char *> getDeviceExtensions();

        void incrementActiveIndex() {
            m_activeIndex = (m_activeIndex + 1) % MAX_FRAMES_IN_FLIGHT;
        }

    private:
#ifdef NDEBUG
        const bool enableValidationLayers = false;
#else
        const bool enableValidationLayers = true;
#endif
        const std::vector<const char *> validationLayers = {
                "VK_LAYER_KHRONOS_validation"};

        void initVulkan();
        void releaseVulkan();

        void createInstance();
        bool checkValidationLayerSupport();
        [[nodiscard]] std::vector<const char *> getRequiredExtensions() const;
        static void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo);
        static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData);
        void setupDebugMessenger();
        static VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo, const VkAllocationCallbacks *pAllocator, VkDebugUtilsMessengerEXT *pDebugMessenger);
        static void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks *pAllocator);
        void pickPhysicalDevice();

        void createLogicalDevice();

        void createCommandPool();
        void createCommandBuffers();

        const uint32_t MAX_FRAMES_IN_FLIGHT = 2;
    };
} // namespace raven