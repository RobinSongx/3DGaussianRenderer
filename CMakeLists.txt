cmake_minimum_required(VERSION 3.18)
project(3DGaussianRenderer LANGUAGES CXX CUDA)

if(MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/utf-8")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/core)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/app)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/glm)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/glfw/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/glew/include)

set(CORE_SOURCES
    core/consts.h
    core/cuda_error_check.h
    core/cuda_error_check.cpp
    core/utilities.h
    core/device_buffer.cu
    core/gaussian_render.cuh
    core/gaussian_render.cu
    core/ply_parser.h
    core/ply_parser.cpp
    core/camera_controls.h
    core/camera_controls.cpp
)

set(APP_SOURCES
    app/main.cpp
)

set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
add_subdirectory(external/glfw)

add_executable(gaussian_render ${CORE_SOURCES} ${APP_SOURCES})

target_link_libraries(gaussian_render
    CUDA::cudart
    OpenGL::GL
    glfw
)

if(WIN32)
    target_link_directories(gaussian_render PRIVATE external/glew/lib/Release/x64)
    target_link_libraries(gaussian_render glew32)
endif()

set_target_properties(gaussian_render PROPERTIES
    CUDA_ARCHITECTURES "native"
)

if(MSVC)
    target_compile_options(gaussian_render PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(gaussian_render PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
endif()
