#pragma once

#include "engine/passes/ComputePass.h"
#include "engine/util/Paths.h"

namespace engine {
    class SingleRadixSortPass : public ComputePass {
    public:
        explicit SingleRadixSortPass(GPUContext *gpuContext) : ComputePass(gpuContext) {
        }

        enum ComputeStage {
            RADIX_SORT = 0,
        };

        struct PushConstants {
            uint32_t g_num_elements;
        };

        PushConstants m_pushConstants{};

    protected:
        std::vector<std::shared_ptr<Shader>> createShaders() override;

        void recordCommands(VkCommandBuffer commandBuffer) override;

        void createPipelineLayouts() override;
    };
} // namespace engine