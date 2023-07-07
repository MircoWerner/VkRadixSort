#pragma once

#include "SingleRadixSortPass.h"

#include <random>
#include <utility>

namespace engine {
    class MultiRadixSort {
    public:
        void execute(GPUContext *gpuContext) {
            // gpu context
            m_gpuContext = gpuContext;

            // compute pass
            m_pass = std::make_shared<SingleRadixSortPass>(gpuContext);
            m_pass->create();
            m_pass->setGlobalInvocationSize(SingleRadixSortPass::RADIX_SORT, 1, 1, 1);

            // uniforms
            m_uniformConstantsRadixSort = m_pass->getUniform(SingleRadixSortPass::RADIX_SORT, 0);
            m_uniformConstantsRadixSort->setVariable<uint>("g_num_elements", NUM_ELEMENTS);
            m_uniformConstantsRadixSort->upload(m_gpuContext->getActiveIndex());

            // buffers
            prepareBuffers();

            // set storage buffers
            m_pass->setStorageBuffer(SingleRadixSortPass::RADIX_SORT, 1, m_buffers[INPUT_BUFFER_INDEX].get());
            m_pass->setStorageBuffer(SingleRadixSortPass::RADIX_SORT, 2, m_buffers[1 - INPUT_BUFFER_INDEX].get());

            // execute pass
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            m_pass->execute(VK_NULL_HANDLE);
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

        std::shared_ptr<SingleRadixSortPass> m_pass;

        std::shared_ptr<Uniform> m_uniformConstantsRadixSort;

        const uint32_t NUM_ELEMENTS = 1000000;

        const uint32_t NUM_ELEMENTS_BYTES = NUM_ELEMENTS * sizeof(uint32_t);

        std::vector<std::shared_ptr<Buffer>> m_buffers = std::vector<std::shared_ptr<Buffer>>(2);

        std::vector<uint32_t> m_elementsIn;

        static const uint32_t INPUT_BUFFER_INDEX = 0;

        static inline const char* PRINT_PREFIX = "[SingleRadixSort] ";

        void prepareBuffers() {
            generateRandomNumbers(m_elementsIn, NUM_ELEMENTS);
            auto settings0 = Buffer::BufferSettings{.m_sizeBytes = NUM_ELEMENTS_BYTES, .m_bufferUsages = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .m_memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .m_name = "radixSort.elementBuffer0"};
            m_buffers[INPUT_BUFFER_INDEX] = Buffer::fillDeviceWithStagingBuffer(m_gpuContext, settings0, m_elementsIn.data());

            std::vector<uint32_t> zeros;
            generateZeros(zeros, NUM_ELEMENTS);
            auto settings1 = Buffer::BufferSettings{.m_sizeBytes = NUM_ELEMENTS_BYTES, .m_bufferUsages = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .m_memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .m_name = "radixSort.elementBuffer1"};
            m_buffers[1 - INPUT_BUFFER_INDEX] = Buffer::fillDeviceWithStagingBuffer(m_gpuContext, settings1, zeros.data());
        }

        void verify(std::vector<uint32_t> &reference) {
            std::vector<uint32_t> data(NUM_ELEMENTS);
            m_buffers[INPUT_BUFFER_INDEX]->downloadWithStagingBuffer(data.data());
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