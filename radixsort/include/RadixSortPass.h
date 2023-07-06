#pragma once

#include "engine/util/Paths.h"
#include "engine/passes/ComputePass.h"

namespace engine {
    class RadixSortPass : public ComputePass {
    public:
        explicit RadixSortPass(GPUContext *gpuContext) : ComputePass(gpuContext) {
        }

        enum ComputeStage {
            RADIX_SORT_HISTOGRAMS = 0,
            RADIX_SORT = 1,
        };

    protected:
        std::vector<std::shared_ptr<Shader>> createShaders() override {
            return {std::make_shared<Shader>(m_gpuContext, Paths::m_resourceDirectoryPath + "/shaders", "radixsort_histograms.comp"),
                    std::make_shared<Shader>(m_gpuContext, Paths::m_resourceDirectoryPath + "/shaders", "radixsort.comp")};
        }

        void recordCommands(VkCommandBuffer commandBuffer) override {
            recordCommandComputeShaderExecution(commandBuffer, RADIX_SORT_HISTOGRAMS);
            VkMemoryBarrier memoryBarrier0{.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask=VK_ACCESS_SHADER_READ_BIT};
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, {}, 1, &memoryBarrier0, 0, nullptr, 0, nullptr);
            recordCommandComputeShaderExecution(commandBuffer, RADIX_SORT);
            VkMemoryBarrier memoryBarrier1{.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask=VK_ACCESS_SHADER_READ_BIT};
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, {}, 1, &memoryBarrier1, 0, nullptr, 0, nullptr);
        }
    };
}