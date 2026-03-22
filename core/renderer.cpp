#include "renderer.h"
#include "cuda_renderer.cuh"
#include <glm/gtx/transform.hpp>
#include <cuda_runtime.h>
#include <glm/gtx/transform.hpp>

namespace gauss_render {

// 从Camera提取预计算相机数据，打包成CUDA核函数需要的结构
CameraData prep_camera_data(const Camera& camera) {
    CameraData result;
    result.position = camera.GetPosition();
    result.view = camera.GetView();
    result.proj = camera.GetProjection();
    result.viewproj = result.proj * result.view;

    // 预计算视场角余切，用于投影计算
    auto fovy = camera.GetFovY();
    auto height = camera.GetHeight();
    auto width = camera.GetWidth();
    auto tan_fovy = glm::tan(fovy * 0.5f);
    auto tan_fovx = tan_fovy * (float)width / (float)height;
    result.fov_cotangent = glm::vec2(1.0f / tan_fovx, 1.0f / tan_fovy);
    result.aspect = (float)width / (float)height;

    // 预计算深度变换参数
    // 将深度从[zNear, zFar]映射到[-1, 1]，用于投影
    result.depth_scale_bias.x = 2.0f / (camera.GetZFar() - camera.GetZNear());
    result.depth_scale_bias.y = -(camera.GetZFar() + camera.GetZNear()) / (camera.GetZFar() - camera.GetZNear());

    return result;
}

// 将高斯模型从CPU上传到GPU显存
// 分配所有需要的缓冲区，设置全局参数
void Renderer::UploadModel(const GaussianModel& model) {
    // 计算需要分配的内存大小
    auto splat_count = model.num_gaussians;
    auto sh_degree = model.sh_degree;
    auto sh_count = (sh_degree + 1) * (sh_degree + 1);

    // 分配GPU缓冲区
    positions_.resizeIfNeeded(splat_count);
    scale_rotation_.resizeIfNeeded(splat_count);
    colors_.resizeIfNeeded(splat_count);
    conics_.resizeIfNeeded(splat_count);
    screen_ellipses_.resizeIfNeeded(splat_count);
    position_xy_clip_.resizeIfNeeded(splat_count);
    position_z_clip_.resizeIfNeeded(splat_count);
    conics_.resizeIfNeeded(splat_count);
    screen_ellipses_.resizeIfNeeded(splat_count);
    sh_.resizeIfNeeded(splat_count * sh_count);

    // 这里应该从GaussianModel拷贝数据到GPU
    // 实际实现待完成...

    // 设置全局参数到CUDA常量内存
    GlobalArgs args;
    args.splat_count = splat_count;
    args.sh_degree = sh_degree;
    args.sh_count = sh_count;
    args.positions = positions_.getPtr();
    args.scale_rotation = scale_rotation_.getPtr();
    args.colors = colors_.getPtr();
    args.conics = conics_.getPtr();
    args.screen_ellipses = screen_ellipses_.getPtr();
    args.position_xy_clip = position_xy_clip_.getPtr();
    args.position_z_clip = position_z_clip_.getPtr();
    args.spherical_harmonics = sh_.getPtr();
    args.tile_range = tile_ranges_.getPtr();

    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(g_global_args, &args, sizeof(GlobalArgs)));

    // 预分配tile列表缓冲区
    // 容量设为高斯数量，实际每个高斯平均覆盖几个tile，足够用了
    // 如果不够，渲染时会动态扩容
    tile_list_capacity_ = splat_count;
    tile_keys_a_.resizeIfNeeded(tile_list_capacity_);
    tile_keys_b_.resizeIfNeeded(tile_list_capacity_);
    tile_values_a_.resizeIfNeeded(tile_list_capacity_);
    tile_values_b_.resizeIfNeeded(tile_list_capacity_);

    TileListArgs tile_list_args;
    tile_list_args.keys = tile_keys_.current();
    tile_list_args.values = tile_values_.current();
    tile_list_args.capacity = tile_list_capacity_;
    SetTileListArgs(&tile_list_args);
}

// 执行一帧完整渲染
// camera: 当前相机参数
// output_buffer: 输出颜色缓冲区（OpenGL PBO）
// stats: 输出各阶段性能统计
void Renderer::Render(const Camera& camera, uint8_t* output_buffer, RenderStats& stats) {
    auto splat_count = positions_.size();
    if (splat_count == 0) {
        return;
    }

    CudaTimer timer;

    // 预计算相机参数并上传到CUDA常量内存
    auto camera_data = prep_camera_data(camera);
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(g_global_args, (void*)&camera_data, sizeof(CameraData), offsetof(GlobalArgs, camera_data), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(g_global_args, &output_buffer, sizeof(uchar4*), offsetof(GlobalArgs, back_buffer), cudaMemcpyHostToDevice));

    // 步骤1：计算球面谐波颜色（根据视角方向）
    EvaluateSphericalHarmonics(timer, splat_count);
    stats.sh_eval_ms = timer.getElapsed();

    // 步骤2：投影高斯到屏幕空间，计算椭圆参数和逆协方差
    EvaluateSplatClipData(timer, splat_count);
    stats.project_ms = timer.getElapsed();

    // 步骤3：构建tile列表，找出每个高斯覆盖哪些tile
    constexpr int build_tile_num_blocks = 16;
    tile_list_size_ = BuildTileList(timer, build_tile_num_blocks, tile_list_capacity_);
    stats.build_tile_ms = timer.getElapsed();

    // 步骤4：对tile列表按深度排序（key高32位是tile索引，低32位是深度）
    // 排序后同一个tile的高斯自然连续且按深度有序
    SortTileList(timer, tile_list_size_, device_temp_storage_, temp_storage_size_, tile_keys_, tile_values_);
    TileListArgs tile_list_args;
    tile_list_args.keys = tile_keys_.current();
    tile_list_args.values = tile_values_.current();
    tile_list_args.size = tile_list_size_;
    tile_list_args.capacity = tile_list_capacity_;
    SetTileListArgs(&tile_list_args);
    stats.sort_ms = timer.getElapsed();

    // 步骤5：计算每个tile在排序数组中的起始和结束索引
    EvaluateTileRange(timer, tile_list_size_);

    // 步骤6：光栅化所有tile，front-to-back alpha混合
    RasterizeTiles(timer);
    stats.raster_ms = timer.getElapsed();

    stats.total_ms = timer.getElapsed();
}

} // namespace gauss_render
