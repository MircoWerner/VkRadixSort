#include "MultiRadixSort.h"

namespace engine {

    void MultiRadixSort::execute(GPUContext *gpuContext) {
        // gpu context
        m_gpuContext = gpuContext;

        // compute pass
        m_pass = std::make_shared<MultiRadixSortPass>(gpuContext);
        m_pass->create();
        const uint NUM_BLOCKS_PER_WORKGROUP = 32;
        uint32_t globalInvocationSize = NUM_ELEMENTS / NUM_BLOCKS_PER_WORKGROUP;
        uint32_t remainder = NUM_ELEMENTS % NUM_BLOCKS_PER_WORKGROUP;
        globalInvocationSize += remainder > 0 ? 1 : 0;
        m_pass->setGlobalInvocationSize(MultiRadixSortPass::RADIX_SORT_HISTOGRAMS, globalInvocationSize, 1, 1);
        m_pass->setGlobalInvocationSize(MultiRadixSortPass::RADIX_SORT, globalInvocationSize, 1, 1);

        // push constants
        const uint NUM_WORKGROUPS = m_pass->getWorkGroupCount(MultiRadixSortPass::RADIX_SORT_HISTOGRAMS).width;
        assert(NUM_WORKGROUPS == m_pass->getWorkGroupCount(MultiRadixSortPass::RADIX_SORT).width);
        m_pass->m_pushConstantsHistogram.g_num_elements = NUM_ELEMENTS;
        m_pass->m_pushConstantsHistogram.g_num_workgroups = NUM_WORKGROUPS;
        m_pass->m_pushConstantsHistogram.g_num_blocks_per_workgroup = NUM_BLOCKS_PER_WORKGROUP;
        m_pass->m_pushConstants.g_num_elements = NUM_ELEMENTS;
        m_pass->m_pushConstants.g_num_workgroups = NUM_WORKGROUPS;
        m_pass->m_pushConstants.g_num_blocks_per_workgroup = NUM_BLOCKS_PER_WORKGROUP;

        // buffers
        prepareBuffers();
        std::cout << PRINT_PREFIX << "Sorting " << NUM_ELEMENTS << " " << (sizeof(m_elementsIn[0]) * 8) << "bit numbers." << std::endl;

        // set storage buffers
        uint32_t activeIndex = m_gpuContext->getActiveIndex();

        // m_buffer0
        m_pass->setStorageBuffer(activeIndex, MultiRadixSortPass::RADIX_SORT_HISTOGRAMS, 0, m_buffers[0].get()); // iteration 0 and 2 (0,0)
        m_pass->setStorageBuffer(activeIndex, MultiRadixSortPass::RADIX_SORT, 0, m_buffers[0].get());            // iteration 0 and 2 (1,0)
        m_pass->setStorageBuffer((activeIndex + 1) % 2, MultiRadixSortPass::RADIX_SORT, 1, m_buffers[0].get());  // iteration 1 and 3 (1,1)

        m_pass->setStorageBuffer((activeIndex + 1) % 2, MultiRadixSortPass::RADIX_SORT_HISTOGRAMS, 0, m_buffers[1].get()); // iteration 1 and 3 (0,0)
        m_pass->setStorageBuffer(activeIndex, MultiRadixSortPass::RADIX_SORT, 1, m_buffers[1].get());                      // iteration 0 and 2 (1,1)
        m_pass->setStorageBuffer((activeIndex + 1) % 2, MultiRadixSortPass::RADIX_SORT, 0, m_buffers[1].get());            // iteration 1 and 3 (1,0)

        m_pass->setStorageBuffer(MultiRadixSortPass::RADIX_SORT_HISTOGRAMS, 1, m_buffers[2].get());
        m_pass->setStorageBuffer(MultiRadixSortPass::RADIX_SORT, 2, m_buffers[2].get());

        // execute pass
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        VkSemaphore awaitBeforeExecution = VK_NULL_HANDLE;
#ifdef SORT_32BIT
        const uint32_t NUM_ITERATIONS = 4;
#else
        const uint32_t NUM_ITERATIONS = 8;
#endif
        for (uint32_t i = 0; i < NUM_ITERATIONS; i++) {
            m_pass->m_pushConstantsHistogram.g_shift = 8 * i;
            m_pass->m_pushConstants.g_shift = 8 * i;
            awaitBeforeExecution = m_pass->execute(awaitBeforeExecution);
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

        //        std::ofstream myfile;
        //        myfile.open("multiradixsort_block_" + std::to_string(NUM_ELEMENTS) + ".csv", std::ios_base::app);
        //        myfile << NUM_ELEMENTS << " " << NUM_BLOCKS_PER_WORKGROUP << " " << std::to_string(gpuSortTime) << " " << std::to_string(cpuSortTime) << std::endl;
    }

    void MultiRadixSort::prepareBuffers() {
        generateRandomNumbers(m_elementsIn, NUM_ELEMENTS);
        //        printBuffer("elements_in", m_elementsIn, NUM_ELEMENTS);
        auto settings0 = Buffer::BufferSettings{.m_sizeBytes = NUM_ELEMENTS_BYTES, .m_bufferUsages = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .m_memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .m_name = "radixSort.elementBuffer0"};
        m_buffers[0] = Buffer::fillDeviceWithStagingBuffer(m_gpuContext, settings0, m_elementsIn.data());

        std::vector<SORT_TYPE> zeros;
        generateZeros(zeros, NUM_ELEMENTS);
        auto settings1 = Buffer::BufferSettings{.m_sizeBytes = NUM_ELEMENTS_BYTES, .m_bufferUsages = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .m_memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .m_name = "radixSort.elementBuffer1"};
        m_buffers[1] = Buffer::fillDeviceWithStagingBuffer(m_gpuContext, settings1, zeros.data());
        auto settings2 = Buffer::BufferSettings{.m_sizeBytes = static_cast<uint32_t>(m_pass->getWorkGroupCount(MultiRadixSortPass::RADIX_SORT_HISTOGRAMS).width * RADIX_SORT_BINS * sizeof(uint32_t)), .m_bufferUsages = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .m_memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .m_name = "radixSort.histogramsBuffer"};
        m_buffers[2] = Buffer::fillDeviceWithStagingBuffer(m_gpuContext, settings2, zeros.data());
    }

    void MultiRadixSort::verify(std::vector<SORT_TYPE> &reference) {
        std::vector<SORT_TYPE> data(NUM_ELEMENTS);
        m_buffers[0]->downloadWithStagingBuffer(data.data());
        //            printBuffer("elements_out", data, NUM_ELEMENTS);
        testSort(reference, data);
    }

    void MultiRadixSort::printBuffer(const std::string &label, std::vector<SORT_TYPE> &buffer, uint32_t numElements) {
        std::cout << label << ":" << std::endl;
        for (uint32_t i = 0; i < numElements; i++) {
            if (i > 0 && i % 16 == 0) {
                std::cout << std::endl;
            }
            std::cout << std::setfill(' ') << std::setw(9) << buffer[i] << " ";
        }
        std::cout << std::endl;
    }

    void MultiRadixSort::releaseBuffers() {
        for (const auto &buffer: m_buffers) {
            buffer->release();
        }
    }

    void MultiRadixSort::generateRandomNumbers(std::vector<SORT_TYPE> &buffer, uint32_t numElements) {
        // https://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution
        std::random_device rd;
        std::mt19937 gen(rd());
#ifdef SORT_32BIT
        std::uniform_int_distribution<uint32_t> distrib(0, 0x0FFFFFFF);
#else
        std::uniform_int_distribution<uint64_t> distrib(0, 0x0FFFFFFFFFFF);
#endif
        for (int i = 0; i < numElements; i++) {
            buffer.push_back(distrib(gen));
        }
    }

    void MultiRadixSort::generateZeros(std::vector<SORT_TYPE> &buffer, uint32_t numElements) {
        for (int i = 0; i < numElements; i++) {
            buffer.push_back(0);
        }
    }

    double MultiRadixSort::sort(std::vector<SORT_TYPE> &buffer) {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        std::sort(buffer.begin(), buffer.end());
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        return (static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) * std::pow(10, -3));
    }

    bool MultiRadixSort::testSort(std::vector<SORT_TYPE> &reference, std::vector<SORT_TYPE> &outBuffer) {
        if (reference.size() != outBuffer.size()) {
            std::cerr << PRINT_PREFIX << "reference.size() != outBuffer.size()" << std::endl;
            throw std::runtime_error("TEST FAILED.");
        }
        for (uint32_t i = 0; i < reference.size(); i++) {
            if (reference[i] != outBuffer[i]) {
                std::cerr << PRINT_PREFIX << reference[i] << " = reference[" << i << "] != outBuffer[" << i << "] = " << outBuffer[i] << std::endl;
                throw std::runtime_error("TEST FAILED.");
            }
        }
        std::cout << PRINT_PREFIX << "Test passed." << std::endl;
        return true;
    }
} // namespace engine