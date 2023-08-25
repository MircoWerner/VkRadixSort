#pragma once

#include <array>
#include <memory>
#include <optional>
#include <vector>
#include <set>
#include <vulkan/vulkan_core.h>

namespace engine {
    class Queues {
    public:
        enum QueueFamilies {
            GRAPHICS_FAMILY = 0x00000001,
            COMPUTE_FAMILY = 0x00000002,
            TRANSFER_FAMILY = 0x00000004,
        };

        enum Queue {
            GRAPHICS = 0,
            COMPUTE = 1,
            TRANSFER = 2,
        };

        explicit Queues(uint32_t requiredQueueFamilies) : m_requiredQueueFamilies(requiredQueueFamilies){};

        struct QueueFamilyIndices {
            std::optional<uint32_t> graphicsFamily;
            std::optional<uint32_t> computeFamily;
            std::optional<uint32_t> transferFamily;

            [[nodiscard]] bool isComplete(uint32_t requiredQueueFamilies) const {
                uint32_t graphics = GRAPHICS_FAMILY & requiredQueueFamilies;
                uint32_t compute = COMPUTE_FAMILY & requiredQueueFamilies;
                uint32_t transfer = TRANSFER_FAMILY & requiredQueueFamilies;
                return (!graphics || graphicsFamily.has_value())
                       && (!compute || computeFamily.has_value())
                       && (!transfer || transferFamily.has_value());
            }

            void generateQueueCreateInfos(std::vector<VkDeviceQueueCreateInfo> *queueCreateInfos, const float *queuePriorities) {
                std::set<uint32_t> uniqueQueueFamilies;
                if (graphicsFamily.has_value()) {
                    uniqueQueueFamilies.emplace(graphicsFamily.value());
                }
                if (computeFamily.has_value()) {
                    uniqueQueueFamilies.emplace(computeFamily.value());
                }
                if (transferFamily.has_value()) {
                    uniqueQueueFamilies.emplace(transferFamily.value());
                }

                for (uint32_t queueFamily: uniqueQueueFamilies) { // queue create infos for all required queues
                    VkDeviceQueueCreateInfo queueCreateInfo{};
                    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                    queueCreateInfo.queueFamilyIndex = queueFamily;
                    queueCreateInfo.queueCount = 1;
                    queueCreateInfo.pQueuePriorities = queuePriorities;
                    queueCreateInfos->push_back(queueCreateInfo);
                }
            }
        };

        QueueFamilyIndices findQueueFamilies(VkPhysicalDevice physicalDevice) const;

        void generateQueueCreateInfos(VkPhysicalDevice physicalDevice, std::vector<VkDeviceQueueCreateInfo> *queueCreateInfos, float *queuePriorities) const;

        void createQueues(VkDevice device, VkPhysicalDevice physicalDevice);

        VkQueue getQueue(Queue queue);

    private:
        uint32_t m_requiredQueueFamilies;
        std::array<VkQueue, 3> m_queues{}; // destroyed implicitly with the device

        [[nodiscard]] bool isFamilyRequired(QueueFamilies queueFamily) const;
    };
} // namespace raven