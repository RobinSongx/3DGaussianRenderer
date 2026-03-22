#include "renderer.h"
#include "cuda_renderer.cuh"

namespace gauss_render {

void Renderer::UploadModel(const GaussianModel& model) {
    // TODO: implement model uploading to GPU
}

void Renderer::Render(const Camera& camera, uint8_t* output_buffer, RenderStats& stats) {
    // 临时输出固定蓝色，验证管线
    int width = camera.GetWidth();
    int height = camera.GetHeight();
    
    // 通过CUDA内核写入设备内存
    ClearScreen(reinterpret_cast<uchar4*>(output_buffer), width, height);

    // 暂时填充统计数据
    stats.sh_eval_ms = 0.0f;
    stats.project_ms = 0.0f;
    stats.build_tile_ms = 0.0f;
    stats.sort_ms = 0.0f;
    stats.raster_ms = 0.0f;
    stats.total_ms = 0.0f;
}

} // namespace gauss_render
