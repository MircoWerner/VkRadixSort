#pragma once

#include "SingleRadixSortPass.h"

#include <random>
#include <utility>

namespace engine {
    class SingleRadixSort {
    public:
        void execute(GPUContext *gpuContext);

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

        void prepareBuffers();

        void verify(std::vector<uint32_t> &reference);

        static void printBuffer(const std::string &label, std::vector<uint32_t> &buffer, uint32_t numElements);

        void releaseBuffers();

        static void generateRandomNumbers(std::vector<uint32_t> &buffer, uint32_t numElements);

        static void generateZeros(std::vector<uint32_t> &buffer, uint32_t numElements);

        static double sort(std::vector<uint32_t> &buffer);

        static bool testSort(std::vector<uint32_t> &reference, std::vector<uint32_t> &outBuffer);
    };
} // namespace engine