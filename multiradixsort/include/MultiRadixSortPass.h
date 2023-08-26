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
            uint32_t g_num_workgroups;
            uint32_t g_num_blocks_per_workgroup;
        };

        PushConstantsHistograms m_pushConstantsHistogram{};

        struct PushConstants {
            uint32_t g_num_elements;
            uint32_t g_shift;
            uint32_t g_num_workgroups;
            uint32_t g_num_blocks_per_workgroup;
        };

        PushConstants m_pushConstants{};

    protected:
        std::vector<std::shared_ptr<Shader>> createShaders() override;

        void recordCommands(VkCommandBuffer commandBuffer) override;

        void createPipelineLayouts() override;
    };
}