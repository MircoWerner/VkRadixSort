cmake_minimum_required(VERSION 3.18)
project(radixsort VERSION 0.1.0 DESCRIPTION "Vulkan Radix Sort" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

add_subdirectory(lib)
add_subdirectory(engine)

option(MAKE_RADIX_SORT_EXAMPLES "Build Vulkan Radix Sort Example." ON)
if (MAKE_RADIX_SORT_EXAMPLES)
	add_subdirectory(radixsort)
endif()