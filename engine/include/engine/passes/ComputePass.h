#pragma once

#include "Pass.h"

namespace engine {
    class ComputePass : public Pass {
    public:
        explicit ComputePass(GPUContext *gpuContext) : Pass(gpuContext) {
        }

        void setGlobalInvocationSize(uint32_t width, uint32_t height, uint32_t depth) {
            VkExtent3D workGroupSizes = {256, 1, 1}; // TODO(Mirco): reflect this
            uint32_t x = (width + workGroupSizes.width - 1) / workGroupSizes.width;
            uint32_t y = (height + workGroupSizes.height - 1) / workGroupSizes.height;
            uint32_t z = (depth + workGroupSizes.depth - 1) / workGroupSizes.depth;
            m_workgroupCount = {x, y, z};
            std::cout << "m_workgroupCount=(" << x << "," << y << "," << z << ")" << std::endl;
        }

        void recordCommandBuffer(VkCommandBuffer commandBuffer) {
            // fill command buffer
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

            if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
                throw std::runtime_error("failed to begin recording command buffer!");
            }

            //
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);

            std::vector<VkDescriptorSet> descriptorSets;
            getDescriptorSets(descriptorSets);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, descriptorSets.size(), descriptorSets.data(), 0, nullptr);

            vkCmdDispatch(commandBuffer, m_workgroupCount.width, m_workgroupCount.height, m_workgroupCount.depth);

            if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
                throw std::runtime_error("failed to record command buffer!");
            }
        }

        VkSemaphore execute(VkSemaphore awaitBeforeExecution) override {
            vkWaitForFences(m_gpuContext->m_device, 1, &m_fences[m_gpuContext->getActiveIndex()], VK_TRUE, UINT64_MAX); // waiting for the previous frame to finish, blocks the CPU
            vkResetFences(m_gpuContext->m_device, 1, &m_fences[m_gpuContext->getActiveIndex()]);

            vkResetCommandBuffer(m_commandBuffers[m_gpuContext->getActiveIndex()], 0);
            recordCommandBuffer(m_commandBuffers[m_gpuContext->getActiveIndex()]);

//            updateUniformBuffer(m_gpuContext->getActiveIndex());

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            if (awaitBeforeExecution) {
                VkSemaphore waitSemaphores[] = {awaitBeforeExecution}; // wait on dependent operation
                VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
                submitInfo.waitSemaphoreCount = 1;
                submitInfo.pWaitSemaphores = waitSemaphores;
                submitInfo.pWaitDstStageMask = waitStages;
            }
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &m_commandBuffers[m_gpuContext->getActiveIndex()];
            VkSemaphore signalSemaphores[] = {m_signalSemaphores[m_gpuContext->getActiveIndex()]}; // is signaled when the command buffer has finished execution
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = signalSemaphores;

            if (vkQueueSubmit(m_gpuContext->m_queues->getQueue(Queues::COMPUTE), 1, &submitInfo, m_fences[m_gpuContext->getActiveIndex()]) != VK_SUCCESS) { // signal fence after the command buffer finished execution
                throw std::runtime_error("Failed to submit compute command buffer!");
            }

            return m_signalSemaphores[m_gpuContext->getActiveIndex()];
        }

    protected:
        uint32_t findQueueFamilyIndex() override {
            Queues::QueueFamilyIndices queueFamilyIndices = m_gpuContext->m_queues->findQueueFamilies(m_gpuContext->m_physicalDevice);
            return queueFamilyIndices.computeFamily.value();
        }

        void createPipeline() override {
            VkPipelineShaderStageCreateInfo shaderStage = m_shaders[0]->generateShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT); // TODO(Mirco): shaders

            VkComputePipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.layout = m_pipelineLayout;
            pipelineInfo.stage = shaderStage;

            if (vkCreateComputePipelines(m_gpuContext->m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create compute pipeline!");
            }
        }

    private:
        VkExtent3D m_workgroupCount = {0, 0, 0};
    };
}