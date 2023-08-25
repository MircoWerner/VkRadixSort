#pragma once

#include "MultiRadixSortPass.h"

#include <random>
#include <utility>

namespace engine {
    class MultiRadixSort {
    public:
        void execute(GPUContext *gpuContext);

    private:
        GPUContext *m_gpuContext;

        std::shared_ptr<MultiRadixSortPass> m_pass;

        const uint32_t RADIX_SORT_BINS = 256;
        const uint32_t NUM_ELEMENTS = 1000000;

        const uint32_t NUM_ELEMENTS_BYTES = NUM_ELEMENTS * sizeof(uint32_t);

        std::vector<std::shared_ptr<Buffer>> m_buffers = std::vector<std::shared_ptr<Buffer>>(3);

        std::vector<uint32_t> m_elementsIn;

        static inline const char *PRINT_PREFIX = "[MultiRadixSort] ";

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