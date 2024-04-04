#include "SingleRadixSort.h"

namespace engine {

    void SingleRadixSort::execute(GPUContext *gpuContext) {
        // gpu context
        m_gpuContext = gpuContext;

        // compute pass
        m_pass = std::make_shared<SingleRadixSortPass>(gpuContext);
        m_pass->create();
        m_pass->setGlobalInvocationSize(SingleRadixSortPass::RADIX_SORT, 256, 1, 1); // WORKGROUP_SIZE defined in single_radixsort.comp, i.e. we just want to launch a single work group

        // push constants
        m_pass->m_pushConstants.g_num_elements = NUM_ELEMENTS;

        // buffers
        prepareBuffers();
        std::cout << PRINT_PREFIX << "Sorting " << NUM_ELEMENTS << " " << (sizeof(m_elementsIn[0]) * 8) << "bit numbers." << std::endl;

        // set storage buffers
        m_pass->setStorageBuffer(SingleRadixSortPass::RADIX_SORT, 0, m_buffers[INPUT_BUFFER_INDEX].get());
        m_pass->setStorageBuffer(SingleRadixSortPass::RADIX_SORT, 1, m_buffers[1 - INPUT_BUFFER_INDEX].get());

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

        //        std::ofstream myfile;
        //        myfile.open("singleradixsort.csv", std::ios_base::app);
        //        myfile << NUM_ELEMENTS << " " << std::to_string(gpuSortTime) << " " << std::to_string(cpuSortTime) << std::endl;
    }

    void SingleRadixSort::prepareBuffers() {
        generateRandomNumbers(m_elementsIn, NUM_ELEMENTS);
        auto settings0 = Buffer::BufferSettings{.m_sizeBytes = NUM_ELEMENTS_BYTES, .m_bufferUsages = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .m_memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .m_name = "radixSort.elementBuffer0"};
        m_buffers[INPUT_BUFFER_INDEX] = Buffer::fillDeviceWithStagingBuffer(m_gpuContext, settings0, m_elementsIn.data());
        // printBuffer("elements_in", m_elementsIn, NUM_ELEMENTS);

        std::vector<SORT_TYPE> zeros;
        generateZeros(zeros, NUM_ELEMENTS);
        auto settings1 = Buffer::BufferSettings{.m_sizeBytes = NUM_ELEMENTS_BYTES, .m_bufferUsages = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .m_memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .m_name = "radixSort.elementBuffer1"};
        m_buffers[1 - INPUT_BUFFER_INDEX] = Buffer::fillDeviceWithStagingBuffer(m_gpuContext, settings1, zeros.data());
    }

    void SingleRadixSort::verify(std::vector<SORT_TYPE> &reference) {
        std::vector<SORT_TYPE> data(NUM_ELEMENTS);
        m_buffers[INPUT_BUFFER_INDEX]->downloadWithStagingBuffer(data.data());
        // printBuffer("elements_out", data, NUM_ELEMENTS);
        testSort(reference, data);
    }

    void SingleRadixSort::printBuffer(const std::string &label, std::vector<SORT_TYPE> &buffer, uint32_t numElements) {
        std::cout << label << ":" << std::endl;
        for (uint32_t i = 0; i < numElements; i++) {
            if (i > 0 && i % 16 == 0) {
                std::cout << std::endl;
            }
            std::cout << std::setfill(' ') << std::setw(9) << buffer[i] << " ";
        }
        std::cout << std::endl;
    }

    void SingleRadixSort::releaseBuffers() {
        for (const auto &buffer: m_buffers) {
            buffer->release();
        }
    }

    void SingleRadixSort::generateRandomNumbers(std::vector<SORT_TYPE> &buffer, uint32_t numElements) {
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
            // buffer.push_back(numElements - i);
        }
    }

    void SingleRadixSort::generateZeros(std::vector<SORT_TYPE> &buffer, uint32_t numElements) {
        for (int i = 0; i < numElements; i++) {
            buffer.push_back(0);
        }
    }

    double SingleRadixSort::sort(std::vector<SORT_TYPE> &buffer) {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        std::sort(buffer.begin(), buffer.end());
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        return (static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) * std::pow(10, -3));
    }

    bool SingleRadixSort::testSort(std::vector<SORT_TYPE> &reference, std::vector<SORT_TYPE> &outBuffer) {
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