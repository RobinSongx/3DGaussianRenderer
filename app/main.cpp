#include "glfw_window.h"
#include "opengl_display.h"
#include "camera_controller.h"
#include "perf_overlay.h"
#include "../core/types.h"
#include "../core/camera.h"
#include "../core/gaussian_model.h"
#include "../core/ply_loader.h"
#include "../core/renderer.h"
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    using namespace gauss_render;

    const int width = kDefaultScreenWidth;
    const int height = kDefaultScreenHeight;

    cudaDeviceProp prop;
    int dev;
    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 0;
    cudaChooseDevice(&dev, &prop);
    cudaGLSetGLDevice(dev);
    cudaGetDeviceProperties(&prop, dev);

    GlfwWindow window(width, height, "3D Gaussian Splatting Renderer");
    OpenglDisplay display(width, height);

    Camera camera;
    camera.SetResolution(width, height);

    CameraController controller(&camera);

    cudaGraphicsResource_t cuda_resource;
    cudaGraphicsGLRegisterBuffer(&cuda_resource, display.GetColorBufferId(), cudaGraphicsRegisterFlagsNone);

    GaussianModel model;
    RenderStats stats;
    Renderer renderer;

    if (argc > 1) {
        std::cout << "Loading PLY file: " << argv[1] << std::endl;
        int num_gaussians = LoadPly(argv[1], model);
        std::cout << "Loaded " << num_gaussians << " gaussians" << std::endl;
        model.PreprocessSh();
        controller.SetBounds(model.GetBoundsMin(), model.GetBoundsMax());
        renderer.UploadModel(model);
    } else {
        std::cerr << "Usage: " << argv[0] << " <input.ply>" << std::endl;
        return 1;
    }

    window.SetMouseMoveCallback([&controller](float dx, float dy) {
        controller.OnMouseMove(dx, dy);
    });

    window.SetScrollCallback([&controller](float dy) {
        controller.OnScroll(dy);
    });

    double last_time = glfwGetTime();
    long frame_count = 0;
    double fps = 0.0;
    double last_fps_update = last_time;
    int num_gaussians_total = model.num_gaussians;

    while (!window.ShouldClose()) {
        double current_time = glfwGetTime();
        float delta_time = static_cast<float>(current_time - last_time);
        last_time = current_time;
        frame_count++;

        if (current_time - last_fps_update >= 0.5) {
            fps = frame_count / (current_time - last_fps_update);
            frame_count = 0;
            last_fps_update = current_time;
        }

        controller.Update(delta_time);

        uchar4* color_ptr = nullptr;
        size_t byte_size;
        cudaGraphicsMapResources(1, &cuda_resource, nullptr);
        cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&color_ptr), &byte_size, cuda_resource);

        RenderStats frame_stats;
        renderer.Render(camera, reinterpret_cast<uint8_t*>(color_ptr), frame_stats);

        stats.sh_eval_ms += frame_stats.sh_eval_ms;
        stats.project_ms += frame_stats.project_ms;
        stats.build_tile_ms += frame_stats.build_tile_ms;
        stats.sort_ms += frame_stats.sort_ms;
        stats.raster_ms += frame_stats.raster_ms;
        stats.total_ms += frame_stats.total_ms;

        DrawPerformanceOverlay(
            reinterpret_cast<uint8_t*>(color_ptr),
            width,
            height,
            num_gaussians_total,
            frame_stats,
            fps);

        cudaGraphicsUnmapResources(1, &cuda_resource, nullptr);

        display.Render();

        window.SwapBuffers();
        window.PollEvents();

        if (glfwGetKey(window.GetHandle(), GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            break;
        }

        while (glfwGetTime() < last_time + 1.0 / 60.0) {
        }
    }

    cudaGraphicsUnregisterResource(cuda_resource);

    double total_time = glfwGetTime() - last_time;
    double avg_total = stats.total_ms / frame_count;
    std::cout << "=== Average timings per frame ===" << std::endl;
    std::cout << "SH eval:     " << stats.sh_eval_ms / frame_count << " ms" << std::endl;
    std::cout << "Projection:  " << stats.project_ms / frame_count << " ms" << std::endl;
    std::cout << "Build tile:  " << stats.build_tile_ms / frame_count << " ms" << std::endl;
    std::cout << "Sort:        " << stats.sort_ms / frame_count << " ms" << std::endl;
    std::cout << "Rasterize:   " << stats.raster_ms / frame_count << " ms" << std::endl;
    std::cout << "Total:       " << avg_total << " ms" << std::endl;
    std::cout << "FPS:         " << 1000.0 / avg_total << std::endl;

    return 0;
}
