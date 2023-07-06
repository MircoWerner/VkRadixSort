#pragma once

#include "engine/util/Paths.h"
#include "engine/passes/ComputePass.h"

namespace engine {
    class RadixSortPass : public ComputePass {
    public:
        explicit RadixSortPass(GPUContext *gpuContext) : ComputePass(gpuContext) {
        }

    protected:
        std::vector<std::shared_ptr<Shader>> createShaders() override {
            return {std::make_shared<Shader>(m_gpuContext, Paths::m_resourceDirectoryPath + "/shaders", "binaryoperation.comp")};
        }
    };
}