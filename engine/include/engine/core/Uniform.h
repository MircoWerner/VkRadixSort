#pragma once

#include "Buffer.h"
#include "GPUContext.h"
#include "SPIRV-Reflect/spirv_reflect.h"
#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace engine {
    class Uniform {
    public:
        explicit Uniform(GPUContext *gpuContext) : m_gpuContext(gpuContext) {
        }

        static std::shared_ptr<Uniform> reflect(GPUContext *gpuContext, const SpvReflectDescriptorBinding &binding) {
            assert(static_cast<VkDescriptorType>(binding.descriptor_type) == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

            auto uniform = std::make_shared<Uniform>(gpuContext);

            uniform->m_name = binding.type_description->type_name;
            std::cout << uniform->m_name << std::endl;
            uniform->m_set = binding.set;
            uniform->m_binding = binding.binding;

            uniform->m_variables.resize(binding.type_description->member_count);

            for (uint32_t i = 0; i < binding.type_description->member_count; i++) {
                //                const auto &member = binding.type_description->members[i];
                const auto &memberBlock = binding.block.members[i];

                auto &variable = uniform->m_variables[i];

                variable.name = memberBlock.name;
                //                variable.type = member.type_flags;
                variable.bytes = memberBlock.size;
                variable.bytesPadded = memberBlock.padded_size;

                variable.data.resize(variable.bytes);

                //                uint32_t bytes;
                //                switch (member.type_flags) {
                //                    case SPV_REFLECT_TYPE_FLAG_INT:
                //                    case SPV_REFLECT_TYPE_FLAG_FLOAT:
                //                        bytes = member.traits.numeric.scalar.width / 8;
                //                        break;
                //                    case SPV_REFLECT_TYPE_FLAG_VECTOR | SPV_REFLECT_TYPE_FLAG_INT:
                //                    case SPV_REFLECT_TYPE_FLAG_VECTOR | SPV_REFLECT_TYPE_FLAG_FLOAT:
                //                        bytes = member.traits.numeric.vector.component_count * member.traits.numeric.scalar.width / 8;
                //                        break;
                //                    case SPV_REFLECT_TYPE_FLAG_MATRIX | SPV_REFLECT_TYPE_FLAG_VECTOR | SPV_REFLECT_TYPE_FLAG_INT:
                //                    case SPV_REFLECT_TYPE_FLAG_MATRIX | SPV_REFLECT_TYPE_FLAG_VECTOR | SPV_REFLECT_TYPE_FLAG_FLOAT:
                //                        bytes = member.traits.numeric.matrix.column_count * member.traits.numeric.matrix.row_count * member.traits.numeric.scalar.width / 8;
                //                        break;
                //                    default:
                //                        throw std::runtime_error("Unsupported uniform type!");
                //                }
                //                variable.bytes = bytes;

                //                std::cout << variable.name << std::endl;
                //                std::cout << bytes << " " << memberBlock.size << " " << memberBlock.padded_size << std::endl;
                //                assert(bytes == memberBlock.size);
            }

            return uniform;
        }

        void create() {
            m_buffers.resize(m_gpuContext->getMultiBufferedCount());
            for (size_t i = 0; i < m_gpuContext->getMultiBufferedCount(); i++) {
                std::stringstream ss;
                ss << "[Uniform]_" << m_name << "[" << i << "]";
                auto settings = Buffer::BufferSettings{.m_sizeBytes = getBytes(), .m_bufferUsages = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, .m_memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, .m_name=ss.str()};
                m_buffers[i] = std::make_shared<Buffer>(m_gpuContext, settings);
            }
        }

        void release() {
            for (size_t i = 0; i < m_gpuContext->getMultiBufferedCount(); i++) {
                m_buffers[i]->release();
            }
        }

        [[nodiscard]] uint32_t getSet() const {
            return m_set;
        }

        [[nodiscard]] uint32_t getBinding() const {
            return m_binding;
        }

        [[nodiscard]] uint32_t getBytes() const {
            uint32_t bytes = 0;
            for (const auto &variable: m_variables) {
                bytes += variable.bytesPadded;
            }
            return bytes;
        }

        void generateBufferInfo(VkDescriptorBufferInfo *bufferInfo, uint32_t index) {
            bufferInfo->buffer = m_buffers[index]->getBuffer();
            bufferInfo->offset = 0;
            bufferInfo->range = m_buffers[index]->getSizeBytes();
        }

        void upload(uint32_t index) {
            uint32_t bytes = getBytes();
            std::vector<char> data(bytes, 0);

            uint32_t pointer = 0;
            for (uint32_t i = 0; i < m_variables.size(); i++) {
                const auto &variable = m_variables[i];
                data.insert(data.begin() + pointer, variable.data.begin(), variable.data.end());
                pointer += variable.bytesPadded;
            }

            m_buffers[index]->updateHostMemory(bytes, data.data());
        }

        template<typename T>
        void setVariable(const std::string &name, T value) {
            UniformVariable *variable = getVariable(name);
            uint32_t bytes = sizeof(T);
            assert(bytes == variable->bytes);
            char *bytePointer = (char *)(&value);
            for (uint32_t i = 0; i < bytes; i++) {
                variable->data[i] = bytePointer[i];
            }
        }


    private:
        struct UniformVariable {
            std::string name;
            //            SpvReflectTypeFlags type;
            uint32_t bytes{};
            uint32_t bytesPadded{};

            std::vector<char> data; // TODO(Mirco): use one long data vector for all variables in the uniform
        };

        uint32_t m_set{};
        uint32_t m_binding{};

        std::string m_name;

        std::vector<UniformVariable> m_variables;

        GPUContext *m_gpuContext;

        std::vector<std::shared_ptr<Buffer>> m_buffers;

        UniformVariable *getVariable(const std::string &name) {
            for (uint32_t i = 0; i < m_variables.size(); i++) {
                if (strcmp(m_variables[i].name.c_str(), name.c_str()) == 0) {
                    return &m_variables[i];
                }
            }
            throw std::runtime_error("Unknown uniform variable " + name + ".");
        }
    };
} // namespace raven