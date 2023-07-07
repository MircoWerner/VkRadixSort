#pragma once

#include "engine/util/Paths.h"
#include "engine/passes/ComputePass.h"

namespace engine {
    class MultiRadixSortPass : public ComputePass {
    public:
        explicit MultiRadixSortPass(GPUContext *gpuContext) : ComputePass(gpuContext) {
        }

        enum ComputeStage {
            RADIX_SORT_HISTOGRAMS = 0,
            RADIX_SORT = 1,
        };

        struct PushConstantsHistograms {
            uint32_t g_num_elements;
            uint32_t g_shift;
        };

        PushConstantsHistograms m_pushConstantsHistogram{};

        struct PushConstants {
            uint32_t g_num_elements;
            uint32_t g_shift;
            uint32_t g_num_workgroups;
        };

        PushConstants m_pushConstants{};

    protected:
        std::vector<std::shared_ptr<Shader>> createShaders() override {
            return {std::make_shared<Shader>(m_gpuContext, Paths::m_resourceDirectoryPath + "/shaders", "multi_radixsort_histograms.comp"),
                    std::make_shared<Shader>(m_gpuContext, Paths::m_resourceDirectoryPath + "/shaders", "multi_radixsort.comp")};
        }

        void recordCommands(VkCommandBuffer commandBuffer) override {
            vkCmdPushConstants(commandBuffer, m_pipelineLayouts[RADIX_SORT_HISTOGRAMS], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsHistograms), &m_pushConstantsHistogram);
            recordCommandComputeShaderExecution(commandBuffer, RADIX_SORT_HISTOGRAMS);
            VkMemoryBarrier memoryBarrier0{.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask=VK_ACCESS_SHADER_READ_BIT};
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, {}, 1, &memoryBarrier0, 0, nullptr, 0, nullptr);

            vkCmdPushConstants(commandBuffer, m_pipelineLayouts[RADIX_SORT], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &m_pushConstants);
            recordCommandComputeShaderExecution(commandBuffer, RADIX_SORT);
            VkMemoryBarrier memoryBarrier1{.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask=VK_ACCESS_SHADER_READ_BIT};
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, {}, 1, &memoryBarrier1, 0, nullptr, 0, nullptr);
        }

        void createPipelineLayouts() override {
            VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.setLayoutCount = m_descriptorSetLayouts.size();
            pipelineLayoutInfo.pSetLayouts = m_descriptorSetLayouts.data();

            // RADIX_SORT_HISTOGRAMS
            VkPushConstantRange pushConstantRange{};
            pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pushConstantRange.offset = 0;
            pushConstantRange.size = sizeof(PushConstantsHistograms);

            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

            if (vkCreatePipelineLayout(m_gpuContext->m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayouts[RADIX_SORT_HISTOGRAMS]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create pipeline layout!");
            }

            // RADIX_SORT
            pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pushConstantRange.offset = 0;
            pushConstantRange.size = sizeof(PushConstants);

            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

            if (vkCreatePipelineLayout(m_gpuContext->m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayouts[RADIX_SORT]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create pipeline layout!");
            }
        }
    };
}