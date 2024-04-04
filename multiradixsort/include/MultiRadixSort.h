#pragma once

#include "MultiRadixSortPass.h"

#include <random>
#include <utility>

namespace engine {
    class MultiRadixSort {
        // chainging this to SORT_64BIT requires changes in the two shaders (redefine data type of elements_* buffers to uint64_t)
#define SORT_32BIT
        // #define SORT_64_BIT

#ifdef SORT_32BIT
#define SORT_TYPE uint32_t
#else
#define SORT_TYPE uint64_t
#endif

    public:
        void execute(GPUContext *gpuContext);

    private:
        GPUContext *m_gpuContext;

        std::shared_ptr<MultiRadixSortPass> m_pass;

        const uint32_t RADIX_SORT_BINS = 256;
        const uint32_t NUM_ELEMENTS = 1000000;

        const uint32_t NUM_ELEMENTS_BYTES = NUM_ELEMENTS * sizeof(SORT_TYPE);

        std::vector<std::shared_ptr<Buffer>> m_buffers = std::vector<std::shared_ptr<Buffer>>(3);

        std::vector<SORT_TYPE> m_elementsIn;

        static inline const char *PRINT_PREFIX = "[MultiRadixSort] ";

        void prepareBuffers();

        void verify(std::vector<SORT_TYPE> &reference);

        static void printBuffer(const std::string &label, std::vector<SORT_TYPE> &buffer, uint32_t numElements);

        void releaseBuffers();

        static void generateRandomNumbers(std::vector<SORT_TYPE> &buffer, uint32_t numElements);

        static void generateZeros(std::vector<SORT_TYPE> &buffer, uint32_t numElements);

        static double sort(std::vector<SORT_TYPE> &buffer);

        static bool testSort(std::vector<SORT_TYPE> &reference, std::vector<SORT_TYPE> &outBuffer);
    };
} // namespace engine