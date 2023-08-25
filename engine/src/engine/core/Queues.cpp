#include "engine/core/Queues.h"

namespace engine {
    Queues::QueueFamilyIndices Queues::findQueueFamilies(VkPhysicalDevice physicalDevice) const {
        QueueFamilyIndices familyIndices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

        for (int i = 0; i < queueFamilyCount; i++) {
            const auto &queueFamily = queueFamilies[i];
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) { // require queue for graphics commands
                familyIndices.graphicsFamily = i;
            }
            if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) { // require queue for compute commands
                familyIndices.computeFamily = i;
            }
            if (queueFamily.queueFlags & VK_QUEUE_TRANSFER_BIT) { // require queue for transfer commands
                familyIndices.transferFamily = i;
            }
            if (familyIndices.isComplete(m_requiredQueueFamilies)) {
                break;
            }
        }

        return familyIndices;
    }

    void Queues::generateQueueCreateInfos(VkPhysicalDevice physicalDevice, std::vector<VkDeviceQueueCreateInfo> *queueCreateInfos, float *queuePriorities) const {
        QueueFamilyIndices familyIndices = findQueueFamilies(physicalDevice);
        familyIndices.generateQueueCreateInfos(queueCreateInfos, queuePriorities);
    }

    void Queues::createQueues(VkDevice device, VkPhysicalDevice physicalDevice) {
        QueueFamilyIndices familyIndices = findQueueFamilies(physicalDevice);
        if (isFamilyRequired(GRAPHICS_FAMILY)) {
            vkGetDeviceQueue(device, familyIndices.graphicsFamily.value(), 0, &m_queues[GRAPHICS]);
        }
        if (isFamilyRequired(COMPUTE_FAMILY)) {
            vkGetDeviceQueue(device, familyIndices.computeFamily.value(), 0, &m_queues[COMPUTE]);
        }
        if (isFamilyRequired(TRANSFER_FAMILY)) {
            vkGetDeviceQueue(device, familyIndices.transferFamily.value(), 0, &m_queues[TRANSFER]);
        }
    }

    VkQueue Queues::getQueue(Queues::Queue queue) {
        return m_queues[queue];
    }

    bool Queues::isFamilyRequired(Queues::QueueFamilies queueFamily) const {
        return m_requiredQueueFamilies & queueFamily;
    }
} // namespace raven