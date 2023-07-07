cmake_minimum_required(VERSION 3.18)
project(radixsort VERSION 0.1.0 DESCRIPTION "Vulkan Radix Sort" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

add_subdirectory(lib)
add_subdirectory(engine)

option(MAKE_MULTI_RADIX_SORT_EXAMPLE "Build Vulkan Multi Radix Sort Example." ON)
if (MAKE_MULTI_RADIX_SORT_EXAMPLE)
	add_subdirectory(multiradixsort)
endif()
option(MAKE_SINGLE_RADIX_SORT_EXAMPLE "Build Vulkan Single Radix Sort Example." ON)
if (MAKE_SINGLE_RADIX_SORT_EXAMPLE)
	add_subdirectory(singleradixsort)
endif()