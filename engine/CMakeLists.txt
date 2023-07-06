cmake_minimum_required(VERSION 3.18)
project(enginecore VERSION 0.1.0 DESCRIPTION "Vulkan Engine Core" LANGUAGES CXX)

set(ENGINE_HEADERS
        include/engine/core/GPUContext.h
        include/engine/core/Queues.h
        include/engine/core/Buffer.h
        include/engine/core/Shader.h
        include/engine/core/Uniform.h
        include/engine/passes/Pass.h
        include/engine/passes/ComputePass.h
        include/engine/util/Paths.h)

set(ENGINECORE_SOURCES
        src/engine/core/GPUContext.cpp
        src/engine/core/Queues.cpp
        src/engine/core/Shader.cpp)

add_library(enginecore STATIC ${ENGINE_HEADERS} ${ENGINECORE_SOURCES})
add_library(enginecore::enginecore ALIAS enginecore)
set_target_properties(enginecore PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(enginecore
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../lib>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        )