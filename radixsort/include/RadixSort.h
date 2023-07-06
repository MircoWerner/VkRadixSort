#pragma once

#include "RadixSortPass.h"

namespace engine {
    class RadixSort {
    public:
        void execute(GPUContext *gpuContext) {
            // gpu context
            m_gpuContext = gpuContext;

            // compute pass
            m_pass = std::make_shared<RadixSortPass>(gpuContext);
            m_pass->create();
            m_pass->setGlobalInvocationSize(NUM_ELEMENTS, 1, 1);

            // uniforms
            m_uniform = m_pass->getUniform(0, 0);

            // buffers
            prepareBuffers();

            // upload
            m_pass->setStorageBuffer(0, 1, m_inputBuf1.get());
            m_pass->setStorageBuffer(0, 2, m_inputBuf2.get());
            m_pass->setStorageBuffer(0, 3, m_outputBuf.get());
            m_uniform->setVariable<uint32_t>("g_elements", NUM_ELEMENTS);
            m_uniform->upload(m_gpuContext->getActiveIndex());

            // execute pass
            m_pass->execute(nullptr);
            vkDeviceWaitIdle(m_gpuContext->m_device);

            // verify result
            downloadData();

            // clean up
            releaseBuffers();
            m_pass->release();
        }

    private:
        GPUContext *m_gpuContext;

        std::shared_ptr<RadixSortPass> m_pass;

        std::shared_ptr<Uniform> m_uniform;

        const uint32_t NUM_ELEMENTS = 2049;
        std::shared_ptr<Buffer> m_inputBuf1;
        std::shared_ptr<Buffer> m_inputBuf2;
        std::shared_ptr<Buffer> m_outputBuf;

        void prepareBuffers() {
            std::vector<uint32_t> data1(NUM_ELEMENTS);
            std::vector<uint32_t> data2(NUM_ELEMENTS);
            for (uint32_t i = 0; i < NUM_ELEMENTS; i++) {
                data1[i] = i;
                data2[i] = 2 * i;
            }
            auto settingsIn1 = Buffer::BufferSettings{.m_sizeBytes = static_cast<uint32_t>(sizeof(uint32_t) * NUM_ELEMENTS), .m_bufferUsages = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .m_memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .m_name="input1"};
            m_inputBuf1 = Buffer::fillDeviceWithStagingBuffer(m_gpuContext, settingsIn1, data1.data());
            auto settingsIn2 = Buffer::BufferSettings{.m_sizeBytes = static_cast<uint32_t>(sizeof(uint32_t) * NUM_ELEMENTS), .m_bufferUsages = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .m_memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .m_name="input2"};
            m_inputBuf2 = Buffer::fillDeviceWithStagingBuffer(m_gpuContext, settingsIn2, data2.data());
            auto settingsOut = Buffer::BufferSettings{.m_sizeBytes = static_cast<uint32_t>(sizeof(uint32_t) * NUM_ELEMENTS), .m_bufferUsages = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .m_memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .m_name="output"};
            m_outputBuf = std::make_shared<Buffer>(m_gpuContext, settingsOut);
        }

        void downloadData() {
            std::vector<uint32_t> data(NUM_ELEMENTS);
            m_outputBuf->downloadWithStagingBuffer(data.data());
            printBuffer("bufferOut", data, NUM_ELEMENTS);
        }

        static void printBuffer(const std::string &label, std::vector<uint32_t> &buffer, uint32_t numElements) {
            std::cout << label << ":" << std::endl;
            for (uint32_t i = 0; i < numElements; i++) {
                std::cout << std::setfill('0') << std::setw(5) << buffer[i] << " ";
                if ((i + 1) % 32 == 0) {
                    std::cout << std::endl;
                }
            }
            std::cout << std::endl;
        }

        void releaseBuffers() {
            m_inputBuf1->release();
            m_inputBuf2->release();
            m_outputBuf->release();
        }
    };
}