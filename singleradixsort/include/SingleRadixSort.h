#pragma once

#include "SingleRadixSortPass.h"

#include <random>
#include <utility>

namespace engine {
    class SingleRadixSort {
        // chainging this to SORT_64BIT requires changes in the shader (redefine data type of elements_* buffers to uint64_t and set ITERATIONS to 8)
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

        std::shared_ptr<SingleRadixSortPass> m_pass;

        std::shared_ptr<Uniform> m_uniformConstantsRadixSort;

        const uint32_t NUM_ELEMENTS = 1000000;

        const uint32_t NUM_ELEMENTS_BYTES = NUM_ELEMENTS * sizeof(SORT_TYPE);

        std::vector<std::shared_ptr<Buffer>> m_buffers = std::vector<std::shared_ptr<Buffer>>(2);

        std::vector<SORT_TYPE> m_elementsIn;

        static const uint32_t INPUT_BUFFER_INDEX = 0;

        static inline const char *PRINT_PREFIX = "[SingleRadixSort] ";

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