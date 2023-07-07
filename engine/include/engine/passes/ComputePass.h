#pragma once

#include "Pass.h"

namespace engine {
    class ComputePass : public Pass {
    public:
        explicit ComputePass(GPUContext *gpuContext) : Pass(gpuContext) {
        }

        void create() override {
            Pass::create();
            m_workGroupCounts.resize(m_shaders.size());
        }

        void setGlobalInvocationSize(uint32_t stageIndex, uint32_t width, uint32_t height, uint32_t depth) {
            VkExtent3D workGroupSize = m_shaders[stageIndex]->getWorkGroupSize();
            VkExtent3D dispatchSize = getDispatchSize(width, height, depth, workGroupSize);
            m_workGroupCounts[stageIndex] = {dispatchSize.width, dispatchSize.height, dispatchSize.depth};
            std::cout << "m_workGroupSize[" << stageIndex << "]=(" << workGroupSize.width << "," << workGroupSize.height << "," << workGroupSize.depth << ")" << std::endl;
            std::cout << "m_workGroupCount[" << stageIndex << "]=(" << dispatchSize.width << "," << dispatchSize.height << "," << dispatchSize.depth << ")" << std::endl;
        }

        static VkExtent3D getDispatchSize(uint32_t width, uint32_t height, uint32_t depth, VkExtent3D workGroupSize) {
            uint32_t x = (width + workGroupSize.width - 1) / workGroupSize.width;
            uint32_t y = (height + workGroupSize.height - 1) / workGroupSize.height;
            uint32_t z = (depth + workGroupSize.depth - 1) / workGroupSize.depth;
            return {x, y, z};
        }

        VkSemaphore execute(VkSemaphore awaitBeforeExecution) override {
            vkWaitForFences(m_gpuContext->m_device, 1, &m_fences[m_gpuContext->getActiveIndex()], VK_TRUE, UINT64_MAX); // waiting for the previous frame to finish, blocks the CPU
            vkResetFences(m_gpuContext->m_device, 1, &m_fences[m_gpuContext->getActiveIndex()]);

            vkResetCommandBuffer(m_commandBuffers[m_gpuContext->getActiveIndex()], 0);
            fillCommandBuffer(m_commandBuffers[m_gpuContext->getActiveIndex()]);

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            if (awaitBeforeExecution != VK_NULL_HANDLE) {
                VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
                submitInfo.waitSemaphoreCount = 1;
                submitInfo.pWaitSemaphores = &awaitBeforeExecution; // wait on dependent operation
                submitInfo.pWaitDstStageMask = waitStages;
            }
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &m_commandBuffers[m_gpuContext->getActiveIndex()];
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &m_signalSemaphores[m_gpuContext->getActiveIndex()]; // is signaled when the command buffer has finished execution

            if (vkQueueSubmit(m_gpuContext->m_queues->getQueue(Queues::COMPUTE), 1, &submitInfo, m_fences[m_gpuContext->getActiveIndex()]) != VK_SUCCESS) { // signal fence after the command buffer finished execution
                throw std::runtime_error("Failed to submit compute command buffer!");
            }

            return m_signalSemaphores[m_gpuContext->getActiveIndex()];
        }

        [[nodiscard]] VkExtent3D getWorkGroupCount(uint32_t stageIndex) {
            return m_workGroupCounts[stageIndex];
        }

    protected:
        uint32_t findQueueFamilyIndex() override {
            Queues::QueueFamilyIndices queueFamilyIndices = m_gpuContext->m_queues->findQueueFamilies(m_gpuContext->m_physicalDevice);
            return queueFamilyIndices.computeFamily.value();
        }

        void createPipelines() override {
            for (uint32_t stageIndex = 0; stageIndex < m_shaders.size(); stageIndex++) {
                const auto &shader = m_shaders[stageIndex];
                VkPipelineShaderStageCreateInfo shaderStage = shader->generateShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT);

                VkComputePipelineCreateInfo pipelineInfo{};
                pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
                pipelineInfo.layout = m_pipelineLayouts[stageIndex];
                pipelineInfo.stage = shaderStage;

                m_pipelines.emplace_back();
                if (vkCreateComputePipelines(m_gpuContext->m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipelines[m_pipelines.size() - 1]) != VK_SUCCESS) {
                    throw std::runtime_error("Failed to create compute pipeline!");
                }
            }
        }

        virtual void recordCommands(VkCommandBuffer commandBuffer) {
            recordCommandComputeShaderExecution(commandBuffer, 0);
        }

        void recordCommandComputeShaderExecution(VkCommandBuffer commandBuffer, uint32_t stageIndex) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[stageIndex]);

            std::vector<VkDescriptorSet> descriptorSets;
            getDescriptorSets(descriptorSets);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayouts[stageIndex], 0, descriptorSets.size(), descriptorSets.data(), 0, nullptr);

            vkCmdDispatch(commandBuffer, m_workGroupCounts[stageIndex].width, m_workGroupCounts[stageIndex].height, m_workGroupCounts[stageIndex].depth);
        }

    private:
        std::vector<VkExtent3D> m_workGroupCounts;

        void fillCommandBuffer(VkCommandBuffer commandBuffer) {
            // fill command buffer
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

            if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
                throw std::runtime_error("failed to begin recording command buffer!");
            }

            recordCommands(commandBuffer);

            if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
                throw std::runtime_error("failed to record command buffer!");
            }
        }
    };
}