#pragma once

#include "MultiRadixSortPass.h"

#include <random>
#include <utility>

namespace engine {
    class MultiRadixSort {
    public:
        void execute(GPUContext *gpuContext) {
            // gpu context
            m_gpuContext = gpuContext;

            // compute pass
            m_pass = std::make_shared<MultiRadixSortPass>(gpuContext);
            m_pass->create();
            m_pass->setGlobalInvocationSize(MultiRadixSortPass::RADIX_SORT_HISTOGRAMS, NUM_ELEMENTS, 1, 1);
            m_pass->setGlobalInvocationSize(MultiRadixSortPass::RADIX_SORT, NUM_ELEMENTS, 1, 1);

            // uniforms
            const uint NUM_WORKGROUPS = m_pass->getWorkGroupCount(MultiRadixSortPass::RADIX_SORT_HISTOGRAMS).width;
            assert(NUM_WORKGROUPS == m_pass->getWorkGroupCount(MultiRadixSortPass::RADIX_SORT).width);
            m_uniformConstantsRadixSortHistograms = m_pass->getUniform(MultiRadixSortPass::RADIX_SORT_HISTOGRAMS, 0);
            m_uniformConstantsRadixSortHistograms->setVariable<uint>("g_num_elements", NUM_ELEMENTS);
            m_uniformConstantsRadixSort = m_pass->getUniform(MultiRadixSortPass::RADIX_SORT, 0);
            m_uniformConstantsRadixSort->setVariable<uint>("g_num_elements", NUM_ELEMENTS);
            m_uniformConstantsRadixSort->setVariable<uint>("g_num_workgroups", NUM_WORKGROUPS);

            // buffers
            prepareBuffers();

            // set storage buffers
            uint32_t activeIndex = m_gpuContext->getActiveIndex();

            m_pass->setStorageBuffer(activeIndex, MultiRadixSortPass::RADIX_SORT_HISTOGRAMS, 1, m_buffers[0].get());
            m_pass->setStorageBuffer((activeIndex + 1) % 2, MultiRadixSortPass::RADIX_SORT_HISTOGRAMS, 1, m_buffers[1].get());

            m_pass->setStorageBuffer(activeIndex, MultiRadixSortPass::RADIX_SORT, 1, m_buffers[0].get());
            m_pass->setStorageBuffer((activeIndex + 1) % 2, MultiRadixSortPass::RADIX_SORT, 1, m_buffers[1].get());
            m_pass->setStorageBuffer(activeIndex, MultiRadixSortPass::RADIX_SORT, 2, m_buffers[1].get());
            m_pass->setStorageBuffer((activeIndex + 1) % 2, MultiRadixSortPass::RADIX_SORT, 2, m_buffers[0].get());

            m_pass->setStorageBuffer(MultiRadixSortPass::RADIX_SORT_HISTOGRAMS, 3, m_buffers[2].get());
            m_pass->setStorageBuffer(MultiRadixSortPass::RADIX_SORT, 3, m_buffers[2].get());

            // execute pass
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            VkSemaphore awaitBeforeExecution = VK_NULL_HANDLE;
            const uint32_t NUM_ITERATIONS = 4;
            for (uint32_t i = 0; i < NUM_ITERATIONS; i++) {
                m_uniformConstantsRadixSortHistograms->setVariable<uint>("g_shift", 8 * i); // TODO: use push constants instead
                m_uniformConstantsRadixSortHistograms->upload(m_gpuContext->getActiveIndex());
                m_uniformConstantsRadixSort->setVariable<uint>("g_shift", 8 * i);
                m_uniformConstantsRadixSort->upload(m_gpuContext->getActiveIndex());
                awaitBeforeExecution = m_pass->execute(awaitBeforeExecution);
//                vkQueueWaitIdle(m_gpuContext->m_queues->getQueue(Queues::COMPUTE));
                m_gpuContext->incrementActiveIndex();
            }
            vkQueueWaitIdle(m_gpuContext->m_queues->getQueue(Queues::COMPUTE));
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            double gpuSortTime = (static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) * std::pow(10, -3));
            std::cout << PRINT_PREFIX << "GPU sort finished in " << gpuSortTime << "[ms]." << std::endl;

            // cpu sorting
            double cpuSortTime = sort(m_elementsIn);
            std::cout << PRINT_PREFIX << "CPU sort finished in " << cpuSortTime << "[ms]." << std::endl;

            // verify result
            verify(m_elementsIn);

            // clean up
            releaseBuffers();
            m_pass->release();
        }

    private:
        GPUContext *m_gpuContext;

        std::shared_ptr<MultiRadixSortPass> m_pass;

        std::shared_ptr<Uniform> m_uniformConstantsRadixSortHistograms;
        std::shared_ptr<Uniform> m_uniformConstantsRadixSort;

        const uint32_t RADIX_SORT_BINS = 256;
        const uint32_t NUM_ELEMENTS = 1000000;

        const uint32_t NUM_ELEMENTS_BYTES = NUM_ELEMENTS * sizeof(uint32_t);

        std::vector<std::shared_ptr<Buffer>> m_buffers = std::vector<std::shared_ptr<Buffer>>(3);

        std::vector<uint32_t> m_elementsIn;

        static inline const char *PRINT_PREFIX = "[MultiRadixSort] ";

        void prepareBuffers() {
            generateRandomNumbers(m_elementsIn, NUM_ELEMENTS);
            auto settings0 = Buffer::BufferSettings{.m_sizeBytes = NUM_ELEMENTS_BYTES, .m_bufferUsages = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .m_memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .m_name = "radixSort.elementBuffer0"};
            m_buffers[0] = Buffer::fillDeviceWithStagingBuffer(m_gpuContext, settings0, m_elementsIn.data());

            std::vector<uint32_t> zeros;
            generateZeros(zeros, NUM_ELEMENTS);
            auto settings1 = Buffer::BufferSettings{.m_sizeBytes = NUM_ELEMENTS_BYTES, .m_bufferUsages = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .m_memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .m_name = "radixSort.elementBuffer1"};
            m_buffers[1] = Buffer::fillDeviceWithStagingBuffer(m_gpuContext, settings1, zeros.data());
            auto settings2 = Buffer::BufferSettings{.m_sizeBytes = static_cast<uint32_t>(m_pass->getWorkGroupCount(MultiRadixSortPass::RADIX_SORT_HISTOGRAMS).width * RADIX_SORT_BINS * sizeof(uint32_t)), .m_bufferUsages = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .m_memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .m_name = "radixSort.histogramsBuffer"};
            m_buffers[2] = Buffer::fillDeviceWithStagingBuffer(m_gpuContext, settings2, zeros.data());
        }

        void verify(std::vector<uint32_t> &reference) {
            std::vector<uint32_t> data(NUM_ELEMENTS);
            m_buffers[0]->downloadWithStagingBuffer(data.data());
            //            printBuffer("elements_out", data, NUM_ELEMENTS);
            testSort(reference, data);
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
            for (const auto &buffer: m_buffers) {
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

        static double sort(std::vector<uint32_t> &buffer) {
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            std::sort(buffer.begin(), buffer.end());
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            return (static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) * std::pow(10, -3));
        }

        static bool testSort(std::vector<uint32_t> &reference, std::vector<uint32_t> &outBuffer) {
            if (reference.size() != outBuffer.size()) {
                std::cerr << PRINT_PREFIX << "reference.size() != outBuffer.size()" << std::endl;
                return false;
            }
            for (uint32_t i = 0; i < reference.size(); i++) {
                if (reference[i] != outBuffer[i]) {
                    std::cerr << PRINT_PREFIX << reference[i] << " = reference[" << i << "] != outBuffer[" << i << "] = " << outBuffer[i] << std::endl;
                    return false;
                }
            }
            std::cout << PRINT_PREFIX << "Test passed." << std::endl;
            return true;
        }
    };
} // namespace engine