#pragma once

#include "engine/core/Buffer.h"
#include "engine/core/GPUContext.h"
#include "engine/core/Shader.h"

#include <vector>
#include <vulkan/vulkan_core.h>

namespace engine {
    class Pass {
    public:
        explicit Pass(GPUContext *gpuContext) : m_gpuContext(gpuContext) {
        }

        virtual VkSemaphore execute(VkSemaphore awaitBeforeExecution) = 0;

        virtual void create() {
            m_shaders = createShaders();
            m_queueFamilyIndex = findQueueFamilyIndex();
            createCommandPool();
            createCommandBuffers();
            createDescriptorSetLayout();
            createDescriptorPool();
            createDescriptorSets();
            m_pipelineLayouts.resize(m_shaders.size());
            createPipelineLayouts();
            createPipelines();
            createSyncObjects();

            createUniforms();
        }

        virtual void release() {
            releaseUniforms();

            releaseSyncObjects();
            vkDestroyDescriptorPool(m_gpuContext->m_device, m_descriptorPool, nullptr);
            for (auto &descriptorSetLayout: m_descriptorSetLayouts) {
                vkDestroyDescriptorSetLayout(m_gpuContext->m_device, descriptorSetLayout, nullptr);
            }
            for (auto &pipeline: m_pipelines) {
                vkDestroyPipeline(m_gpuContext->m_device, pipeline, nullptr);
            }
            for (auto &pipelineLayout : m_pipelineLayouts) {
                vkDestroyPipelineLayout(m_gpuContext->m_device, pipelineLayout, nullptr);
            }
            vkDestroyCommandPool(m_gpuContext->m_device, m_commandPool, nullptr);
            for (auto &shader: m_shaders) {
                shader->release();
            }
        }

        void setStorageBuffer(uint32_t set, uint32_t binding, Buffer *buffer) {
            uint32_t setIdx = m_descriptorSetToIndex[set];

            Shader::DescriptorSetLayoutData &layout = m_descriptorSetLayoutData[setIdx];
            const auto &bindingLayout = layout.bindings[layout.bindingToIndex[binding]];
            assert(set == layout.set_number);
            assert(binding == bindingLayout.binding);

            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = buffer->getBuffer();
            bufferInfo.offset = 0;
            bufferInfo.range = buffer->getSizeBytes();

            VkWriteDescriptorSet writeDescriptorSet{};
            writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSet.dstBinding = binding;
            writeDescriptorSet.dstArrayElement = 0;
            writeDescriptorSet.descriptorType = bindingLayout.descriptorType;
            writeDescriptorSet.descriptorCount = 1;
            writeDescriptorSet.pBufferInfo = &bufferInfo;

            for (size_t i = 0; i < m_gpuContext->getMultiBufferedCount(); i++) {
                writeDescriptorSet.dstSet = m_descriptorSets[i][setIdx];
                vkUpdateDescriptorSets(m_gpuContext->m_device, 1, &writeDescriptorSet, 0, nullptr);
            }
        }

        void setStorageBuffer(uint32_t multiBufferedIndex, uint32_t set, uint32_t binding, Buffer *buffer) {
            uint32_t setIdx = m_descriptorSetToIndex[set];

            Shader::DescriptorSetLayoutData &layout = m_descriptorSetLayoutData[setIdx];
            const auto &bindingLayout = layout.bindings[layout.bindingToIndex[binding]];
            assert(set == layout.set_number);
            assert(binding == bindingLayout.binding);

            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = buffer->getBuffer();
            bufferInfo.offset = 0;
            bufferInfo.range = buffer->getSizeBytes();

            VkWriteDescriptorSet writeDescriptorSet{};
            writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSet.dstBinding = binding;
            writeDescriptorSet.dstArrayElement = 0;
            writeDescriptorSet.descriptorType = bindingLayout.descriptorType;
            writeDescriptorSet.descriptorCount = 1;
            writeDescriptorSet.pBufferInfo = &bufferInfo;

            writeDescriptorSet.dstSet = m_descriptorSets[multiBufferedIndex][setIdx];
            vkUpdateDescriptorSets(m_gpuContext->m_device, 1, &writeDescriptorSet, 0, nullptr);
        }

        std::shared_ptr<Uniform> getUniform(uint32_t set, uint32_t binding) {
            return m_uniforms[set][binding];
        }

    protected:
        GPUContext *m_gpuContext;

        std::vector<VkPipelineLayout> m_pipelineLayouts{};
        std::vector<VkPipeline> m_pipelines{};

        VkCommandPool m_commandPool{};
        std::vector<VkCommandBuffer> m_commandBuffers; // destroyed implicitly with the command pool

        VkDescriptorPool m_descriptorPool{};
        std::map<uint32_t, uint32_t> m_descriptorSetToIndex;                    // m_descriptorSetToIndex[setId] - mapping of set number to index in the following vectors
        std::vector<Shader::DescriptorSetLayoutData> m_descriptorSetLayoutData; // m_descriptorSetLayoutData[m_descriptorSetToIndex[setId]] - reflected bindings
        std::vector<VkDescriptorSetLayout> m_descriptorSetLayouts{};            // m_descriptorSetLayouts[m_descriptorSetToIndex[setId]]
        std::vector<std::vector<VkDescriptorSet>> m_descriptorSets;             // m_descriptorSets[multibufferedId][m_descriptorSetToIndex[setId]] - destroyed implicitly with the descriptor pool

        std::map<uint32_t, std::map<uint32_t, std::shared_ptr<Uniform>>> m_uniforms; // m_uniforms[setId][bindingId]

        std::vector<std::shared_ptr<Shader>> m_shaders;

        // synchronization
        std::vector<VkSemaphore> m_signalSemaphores;
        std::vector<VkFence> m_fences;

        uint32_t m_queueFamilyIndex{};

        virtual uint32_t findQueueFamilyIndex() = 0;

        virtual void createPipelines() = 0;

        virtual void createPipelineLayouts() {
            VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.setLayoutCount = m_descriptorSetLayouts.size();
            pipelineLayoutInfo.pSetLayouts = m_descriptorSetLayouts.data();
            pipelineLayoutInfo.pushConstantRangeCount = 0;
            pipelineLayoutInfo.pPushConstantRanges = nullptr;

            for (uint32_t stageIndex = 0; stageIndex < m_shaders.size(); stageIndex++) {
                if (vkCreatePipelineLayout(m_gpuContext->m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayouts[stageIndex]) != VK_SUCCESS) {
                    throw std::runtime_error("Failed to create pipeline layout!");
                }
            }
        };

        virtual std::vector<std::shared_ptr<Shader>> createShaders() = 0;

        void getDescriptorSets(std::vector<VkDescriptorSet> &sets) {
            sets.resize(m_descriptorSets[m_gpuContext->getActiveIndex()].size());
            for (uint32_t i = 0; i < m_descriptorSets[m_gpuContext->getActiveIndex()].size(); i++) {
                sets[i] = m_descriptorSets[m_gpuContext->getActiveIndex()][i];
            }
        }

    private:
        void createCommandPool() {
            VkCommandPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            poolInfo.queueFamilyIndex = m_queueFamilyIndex;

            if (vkCreateCommandPool(m_gpuContext->m_device, &poolInfo, nullptr, &m_commandPool) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create command pool!");
            }
        }

        void createCommandBuffers() {
            m_commandBuffers.resize(m_gpuContext->getMultiBufferedCount());

            VkCommandBufferAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.commandPool = m_commandPool;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = (uint32_t) m_commandBuffers.size();

            if (vkAllocateCommandBuffers(m_gpuContext->m_device, &allocInfo, m_commandBuffers.data()) != VK_SUCCESS) {
                throw std::runtime_error("Failed to allocate command buffers!");
            }
        }

        void createDescriptorSetLayout() {
            // retrieve information using SPIRV-reflect
            for (const auto &shader: m_shaders) {                                     // for every shader
                for (const auto &layoutData: *shader->getDescriptorSetLayoutData()) { // retrieve all sets
                    uint32_t setId = layoutData.first;
                    const Shader::DescriptorSetLayoutData &layout = layoutData.second;

                    if (m_descriptorSetToIndex.contains(setId)) {
                        // merge with existing set
                        uint32_t index = m_descriptorSetToIndex[setId];
                        auto &mergedLayoutData = m_descriptorSetLayoutData[index];
                        assert(setId == mergedLayoutData.set_number);
                        mergedLayoutData.bindings.insert(mergedLayoutData.bindings.end(), layout.bindings.begin(), layout.bindings.end());
                        //                        for (auto &binding : layout.bindings) {
                        //
                        //                        }
                    } else {
                        // insert new set
                        uint32_t index = m_descriptorSetLayoutData.size();
                        m_descriptorSetToIndex[setId] = index;
                        Shader::DescriptorSetLayoutData newLayoutData{.set_number = setId, .bindings = layout.bindings};
                        m_descriptorSetLayoutData.push_back(newLayoutData);
                    }
                }

                for (const auto &[bindingId, uniforms]: *shader->getUniforms()) {
                    for (const auto &uniform: uniforms) {
                        m_uniforms[uniform->getSet()][uniform->getBinding()] = uniform;
                    }
                }
            }

            for (auto &layoutData: m_descriptorSetLayoutData) {
                layoutData.create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                layoutData.create_info.bindingCount = layoutData.bindings.size();
                layoutData.create_info.pBindings = layoutData.bindings.data();

                layoutData.generateBindingToIndexMap();
            }

            m_descriptorSetLayouts.resize(m_descriptorSetLayoutData.size());
            for (auto &layoutData: m_descriptorSetLayoutData) {
                uint32_t index = m_descriptorSetToIndex[layoutData.set_number];
                if (vkCreateDescriptorSetLayout(m_gpuContext->m_device, &layoutData.create_info, nullptr, &m_descriptorSetLayouts[index]) != VK_SUCCESS) {
                    throw std::runtime_error("Failed to create descriptor set layout!");
                }
            }
        }

        void createDescriptorPool() {
            // retrieve information using SPIRV-reflect
            std::map<VkDescriptorType, uint32_t> descriptorCounts;
            for (const auto &layoutData: m_descriptorSetLayoutData) {
                for (const auto &binding: layoutData.bindings) {
                    if (descriptorCounts.contains(binding.descriptorType)) {
                        descriptorCounts[binding.descriptorType] += binding.descriptorCount;
                    } else {
                        descriptorCounts[binding.descriptorType] = binding.descriptorCount;
                    }
                }
            }

            std::vector<VkDescriptorPoolSize> poolSizes;
            for (const auto &[descriptorType, count]: descriptorCounts) {
                poolSizes.emplace_back(descriptorType, count * m_gpuContext->getMultiBufferedCount());
            }

            VkDescriptorPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
            poolInfo.pPoolSizes = poolSizes.data();
            poolInfo.maxSets = m_gpuContext->getMultiBufferedCount() * m_descriptorSetLayouts.size();

            if (vkCreateDescriptorPool(m_gpuContext->m_device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create descriptor pool!");
            }
        }

        void createDescriptorSets() {
            // allocate descriptor sets (valid values will be set later)
            m_descriptorSets.resize(m_gpuContext->getMultiBufferedCount());
            for (uint32_t i = 0; i < m_gpuContext->getMultiBufferedCount(); i++) {
                VkDescriptorSetAllocateInfo allocInfo{};
                allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                allocInfo.descriptorPool = m_descriptorPool;
                allocInfo.descriptorSetCount = m_descriptorSetLayouts.size();
                allocInfo.pSetLayouts = m_descriptorSetLayouts.data();

                m_descriptorSets[i].resize(m_descriptorSetLayouts.size());
                if (vkAllocateDescriptorSets(m_gpuContext->m_device, &allocInfo, m_descriptorSets[i].data()) != VK_SUCCESS) {
                    throw std::runtime_error("Failed to allocate descriptor sets!");
                }
            }
        }

        void createSyncObjects() {
            m_signalSemaphores.resize(m_gpuContext->getMultiBufferedCount());
            m_fences.resize(m_gpuContext->getMultiBufferedCount());

            /*
             * VkSemaphoreTypeCreateInfo timelineCreateInfo;
timelineCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
timelineCreateInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
timelineCreateInfo.initialValue = 0;
VkSemaphoreCreateInfo semaphoreInfo{};
semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
semaphoreInfo.pNext = &timelineCreateInfo;
semaphoreInfo.flags = 0;
             */
            VkSemaphoreCreateInfo semaphoreInfo{};
            semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

            VkFenceCreateInfo fenceInfo{};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // create in signaled state, such that the first vkWaitForFences(..) call immediately returns (there is no previous frame to wait for when starting the application)

            for (size_t i = 0; i < m_gpuContext->getMultiBufferedCount(); i++) {
                if (vkCreateSemaphore(m_gpuContext->m_device, &semaphoreInfo, nullptr, &m_signalSemaphores[i]) != VK_SUCCESS ||
                    vkCreateFence(m_gpuContext->m_device, &fenceInfo, nullptr, &m_fences[i]) != VK_SUCCESS) {
                    throw std::runtime_error("Failed to create synchronization objects for a frame!");
                }
            }
        }

        void releaseSyncObjects() {
            for (size_t i = 0; i < m_gpuContext->getMultiBufferedCount(); i++) {
                vkDestroySemaphore(m_gpuContext->m_device, m_signalSemaphores[i], nullptr);
                vkDestroyFence(m_gpuContext->m_device, m_fences[i], nullptr);
            }
        }

        void createUniforms() {
            for (const auto &[set, uniforms]: m_uniforms) {
                for (const auto &[binding, uniform]: uniforms) {
                    uniform->create();
                    setUniform(uniform.get());
                }
            }
        }

        void releaseUniforms() {
            for (const auto &[set, uniforms]: m_uniforms) {
                for (const auto &[binding, uniform]: uniforms) {
                    uniform->release();
                }
            }
        }

        void setUniform(Uniform *uniform) {
            uint32_t setIdx = m_descriptorSetToIndex[uniform->getSet()];

            Shader::DescriptorSetLayoutData &layout = m_descriptorSetLayoutData[setIdx];
            const auto &bindingLayout = layout.bindings[layout.bindingToIndex[uniform->getBinding()]];

            assert(uniform->getSet() == layout.set_number);
            assert(uniform->getBinding() == bindingLayout.binding);

            VkWriteDescriptorSet writeDescriptorSet{};
            writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSet.dstBinding = uniform->getBinding();
            writeDescriptorSet.dstArrayElement = 0;
            writeDescriptorSet.descriptorType = bindingLayout.descriptorType;
            writeDescriptorSet.descriptorCount = 1;

            for (uint32_t i = 0; i < m_gpuContext->getMultiBufferedCount(); i++) {
                VkDescriptorBufferInfo bufferInfo{};
                uniform->generateBufferInfo(&bufferInfo, i);

                writeDescriptorSet.dstSet = m_descriptorSets[i][setIdx];
                writeDescriptorSet.pBufferInfo = &bufferInfo;
                vkUpdateDescriptorSets(m_gpuContext->m_device, 1, &writeDescriptorSet, 0, nullptr);
            }
        }
    };
} // namespace engine