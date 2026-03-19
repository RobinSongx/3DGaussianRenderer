cmake_minimum_required(VERSION 3.18)
project(3DGaussianRenderer LANGUAGES CXX CUDA)

# 解决路径空格问题
if(MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/utf-8")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)

# 设置输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/core)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/app)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/glm)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/glfw/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/glew/include)

# 核心源文件
set(CORE_SOURCES
    core/types.h
    core/camera.h
    core/camera.cpp
    core/gaussian_model.h
    core/gaussian_model.cpp
    core/ply_loader.h
    core/ply_loader.cpp
    core/device_buffer.h
    core/cuda_utils.h
    core/renderer.h
    core/renderer.cpp
    core/cuda_renderer.cuh
    core/cuda_renderer.cu
)

# 应用源文件
set(APP_SOURCES
    app/main.cpp
    app/camera_controller.h
    app/camera_controller.cpp
    app/glfw_window.h
    app/glfw_window.cpp
    app/opengl_display.h
    app/opengl_display.cpp
    app/perf_overlay.h
    app/perf_overlay.cpp
)

# GLFW
add_subdirectory(external/glfw)

# 可执行文件
add_executable(gaussian_render ${CORE_SOURCES} ${APP_SOURCES})

# 链接库
target_link_libraries(gaussian_render
    CUDA::cudart
    CUDA::cublas
    OpenGL::GL
    glfw
)

# GLEW
if(WIN32)
    target_link_directories(gaussian_render PRIVATE external/glew/lib/Release/x64)
    target_link_libraries(gaussian_render glew32)
endif()

# 设置CUDA架构，根据你的GPU调整
set_target_properties(gaussian_render PROPERTIES
    CUDA_ARCHITECTURES "native"
)

# 开启编译警告
if(MSVC)
    target_compile_options(gaussian_render PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(gaussian_render PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
endif()
