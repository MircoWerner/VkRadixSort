cmake_minimum_required(VERSION 3.18)
project(singleradixsortexample VERSION 0.1.0 DESCRIPTION "Vulkan Radix Sort Example" LANGUAGES CXX)

set(PROJECT_HEADERS
        include/SingleRadixSort.h
        include/SingleRadixSortPass.h)

set(PROJECT_SOURCES
        src/bin/SingleRadixSortExample.cpp
        src/SingleRadixSort.cpp
        src/SingleRadixSortPass.cpp
)

add_executable(singleradixsortexample ${PROJECT_HEADERS} ${PROJECT_SOURCES})

target_link_libraries(singleradixsortexample Vulkan::Vulkan enginecore spirv-reflect)

target_include_directories(singleradixsortexample
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        )

SET(RESOURCE_DIRECTORY_PATH \"${CMAKE_CURRENT_SOURCE_DIR}/resources\")
if (RESOURCE_DIRECTORY_PATH)
    target_compile_definitions(singleradixsortexample PRIVATE RESOURCE_DIRECTORY_PATH=${RESOURCE_DIRECTORY_PATH})
endif()