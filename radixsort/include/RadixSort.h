#pragma once

#include "RadixSortPass.h"

#include <random>
#include <utility>

namespace engine {
    class RadixSort {
    public:
        void execute(GPUContext *gpuContext) {
            // gpu context
            m_gpuContext = gpuContext;

            // compute pass
            m_pass = std::make_shared<RadixSortPass>(gpuContext);
            m_pass->create();
            m_pass->setGlobalInvocationSize(RadixSortPass::RADIX_SORT_HISTOGRAMS, NUM_ELEMENTS, 1, 1);
            m_pass->setGlobalInvocationSize(RadixSortPass::RADIX_SORT, NUM_ELEMENTS, 1, 1);

            // uniforms
            const uint NUM_WORKGROUPS = m_pass->getWorkGroupCount(RadixSortPass::RADIX_SORT_HISTOGRAMS).width;
            assert(NUM_WORKGROUPS == m_pass->getWorkGroupCount(RadixSortPass::RADIX_SORT).width);
            m_uniformConstantsRadixSortHistograms = m_pass->getUniform(RadixSortPass::RADIX_SORT_HISTOGRAMS, 0);
            m_uniformConstantsRadixSortHistograms->setVariable<uint>("g_num_elements", NUM_ELEMENTS);
            m_uniformConstantsRadixSort = m_pass->getUniform(RadixSortPass::RADIX_SORT, 0);
            m_uniformConstantsRadixSort->setVariable<uint>("g_num_elements", NUM_ELEMENTS);
            m_uniformConstantsRadixSort->setVariable<uint>("g_num_workgroups", NUM_WORKGROUPS);

            // buffers
            prepareBuffers();

            // execute pass
            const uint32_t NUM_ITERATIONS = 4;
            for (uint32_t i = 0; i < NUM_ITERATIONS; i++) {
                m_pass->setStorageBuffer(RadixSortPass::RADIX_SORT_HISTOGRAMS, 1, m_buffers[m_inputBufferIndex].get());
                m_pass->setStorageBuffer(RadixSortPass::RADIX_SORT_HISTOGRAMS, 3, m_buffers[2].get());
                m_pass->setStorageBuffer(RadixSortPass::RADIX_SORT, 1, m_buffers[m_inputBufferIndex].get());
                m_pass->setStorageBuffer(RadixSortPass::RADIX_SORT, 2, m_buffers[1 - m_inputBufferIndex].get());
                m_pass->setStorageBuffer(RadixSortPass::RADIX_SORT, 3, m_buffers[2].get());
                m_uniformConstantsRadixSortHistograms->setVariable<uint>("g_shift", 8 * i);
                m_uniformConstantsRadixSortHistograms->upload(m_gpuContext->getActiveIndex());
                m_uniformConstantsRadixSort->setVariable<uint>("g_shift", 8 * i);
                m_uniformConstantsRadixSort->upload(m_gpuContext->getActiveIndex());
                m_pass->execute({});
                vkDeviceWaitIdle(m_gpuContext->m_device);
                m_inputBufferIndex = 1 - m_inputBufferIndex;
            }
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

        std::shared_ptr<Uniform> m_uniformConstantsRadixSortHistograms;
        std::shared_ptr<Uniform> m_uniformConstantsRadixSort;

        const uint32_t RADIX_SORT_BINS = 256;
        const uint32_t NUM_ELEMENTS = 12000;

        const uint32_t NUM_ELEMENTS_BYTES = NUM_ELEMENTS * sizeof(uint32_t);

        std::vector<std::shared_ptr<Buffer>> m_buffers = std::vector<std::shared_ptr<Buffer>>(3);
        uint32_t m_inputBufferIndex = 0;

        std::vector<uint32_t> m_elementsIn;

        void prepareBuffers() {
            generateRandomNumbers(m_elementsIn, NUM_ELEMENTS);
            auto settings0 = Buffer::BufferSettings{.m_sizeBytes = NUM_ELEMENTS_BYTES, .m_bufferUsages = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .m_memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .m_name = "radixSort.elementBuffer0"};
            m_buffers[0] = Buffer::fillDeviceWithStagingBuffer(m_gpuContext, settings0, m_elementsIn.data());

            std::vector<uint32_t> zeros;
            generateZeros(zeros, NUM_ELEMENTS);
            auto settings1 = Buffer::BufferSettings{.m_sizeBytes = NUM_ELEMENTS_BYTES, .m_bufferUsages = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .m_memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .m_name = "radixSort.elementBuffer1"};
            m_buffers[1] = Buffer::fillDeviceWithStagingBuffer(m_gpuContext, settings1, zeros.data());
            auto settings2 = Buffer::BufferSettings{.m_sizeBytes = static_cast<uint32_t>(m_pass->getWorkGroupCount(RadixSortPass::RADIX_SORT_HISTOGRAMS).width * RADIX_SORT_BINS * sizeof(uint32_t)), .m_bufferUsages = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .m_memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .m_name = "radixSort.histogramsBuffer"};
            m_buffers[2] = Buffer::fillDeviceWithStagingBuffer(m_gpuContext, settings2, zeros.data());
        }

        void downloadData() {
            std::vector<uint32_t> data(NUM_ELEMENTS);
            m_buffers[m_inputBufferIndex]->downloadWithStagingBuffer(data.data());
            printBuffer("elements_out", data, NUM_ELEMENTS);
        }

        static void printBuffer(const std::string &label, std::vector<uint32_t> &buffer, uint32_t numElements) {
            std::cout << label << ":" << std::endl;
            for (uint32_t i = 0; i < numElements; i++) {
                if (i > 0 && i % 16 == 0) {
                    std::cout << std::endl;
                }
                std::cout << std::setfill('0') << std::setw(9) << buffer[i] << " ";
            }
            std::cout << std::endl;
        }

        void releaseBuffers() {
            for (const auto &buffer : m_buffers) {
                buffer->release();
            }
        }

        static void generateRandomNumbers(std::vector<uint32_t> &buffer, uint32_t numElements) {
            // https://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> distrib(0, 0x0FFFFFFF);
            for (int i = 0; i < numElements; i++) {
                buffer.push_back(distrib(gen));
            }
        }

        static void generateZeros(std::vector<uint32_t> &buffer, uint32_t numElements) {
            for (int i = 0; i < numElements; i++) {
                buffer.push_back(0);
            }
        }
    };
} // namespace engine