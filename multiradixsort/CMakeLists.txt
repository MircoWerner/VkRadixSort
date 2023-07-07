cmake_minimum_required(VERSION 3.18)
project(multiradixsortexample VERSION 0.1.0 DESCRIPTION "Vulkan Radix Sort Example" LANGUAGES CXX)

set(PROJECT_HEADERS
        include/MultiRadixSort.h
        include/MultiRadixSortPass.h)

set(PROJECT_SOURCES
        src/bin/MultiRadixSortExample.cpp
        )

add_executable(multiradixsortexample ${PROJECT_HEADERS} ${PROJECT_SOURCES})

target_link_libraries(multiradixsortexample Vulkan::Vulkan enginecore spirv-reflect)

target_include_directories(multiradixsortexample
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        )

SET(RESOURCE_DIRECTORY_PATH \"${CMAKE_CURRENT_SOURCE_DIR}/resources\")
if (RESOURCE_DIRECTORY_PATH)
    target_compile_definitions(multiradixsortexample PRIVATE RESOURCE_DIRECTORY_PATH=${RESOURCE_DIRECTORY_PATH})
endif()