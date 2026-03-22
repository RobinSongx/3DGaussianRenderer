#include "renderer.h"
#include "cuda_renderer.cuh"
#include <glm/gtx/transform.hpp>
#include <cuda_runtime.h>
#include <glm/gtx/transform.hpp>

namespace gauss_render {

// 预计算相机数据，传递给CUDA
CameraData prep_camera_data(const Camera& camera) {
    CameraData result;
    result.position = camera.Position();
    result.view = camera.GetViewMatrix();
    result.proj = camera.GetProjectionMatrix();
    result.viewproj = result.proj * result.view;

    // 预计算fov相关参数
    auto fovy = camera.GetFovY();
    auto height = camera.GetHeight();
    auto width = camera.GetWidth();
    auto tan_fovy = glm::tan(fovy * 0.5f);
    auto tan_fovx = tan_fovy * (float)width / (float)height;
    result.fov_cotangent = glm::vec2(1.0f / tan_fovx, 1.0f / tan_fovy);
    result.aspect = (float)width / (float)height;

    // 计算深度变换参数：proj[2][2] = 2 / (zFar - zNear); proj[3][2] = -(zFar + zNear) / (zFar - zNear);
    // 所以 depth = (proj[2][2] * z + proj[3][2]) / (-z) = 2/(zFar - zNear) + (zFar + zNear)/(z(zFar - zNear)) = ...
    // 这里我们预计算，方便投影：
    result.depth_scale_bias.x = 2.0f / (camera.GetZFar() - camera.GetZNear());
    result.depth_scale_bias.y = -(camera.GetZFar() + camera.GetZNear()) / (camera.GetZFar() - camera.GetZNear());

    return result;
}

void Renderer::UploadModel(const GaussianModel& model) {
    // 计算需要分配的内存
    auto splat_count = model.num_gaussians;
    auto sh_degree = model.max_sh_degree;
    auto sh_count = (sh_degree + 1) * (sh_degree + 1);

    // 分配GPU缓冲区
    positions_.Resize(splat_count);
    scale_rotation_.Resize(splat_count);
    colors_.Resize(splat_count);
    conics_.Resize(splat_count);
    screen_ellipses_.Resize(splat_count);
    position_xy_clip_.Resize(splat_count);
    position_z_clip_.Resize(splat_count);
    conics_.Resize(splat_count);
    screen_ellipses_.Resize(splat_count);
    sh_.Resize(splat_count * sh_count);

    // 拷贝数据到GPU
    positions_.CopyFromHost(model.positions.data());
    scale_rotation_.CopyFromHost(model.scale_rotation.data());
    sh_.CopyFromHost(model.shs.data());

    // 设置全局参数到CUDA常量内存
    GlobalArgs args;
    args.splat_count = splat_count;
    args.sh_degree = sh_degree;
    args.sh_count = sh_count;
    args.positions = positions_.Ptr();
    args.scale_rotation = scale_rotation_.Ptr();
    args.colors = colors_.Ptr();
    args.conics = conics_.Ptr();
    args.screen_ellipses = screen_ellipses_.Ptr();
    args.position_xy_clip = position_xy_clip_.Ptr();
    args.position_z_clip = position_z_clip_.Ptr();
    args.spherical_harmonics = sh_.Ptr();
    args.tile_range = tile_ranges_.Ptr();

    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(g_global_args, &args, sizeof(GlobalArgs)));

    // 预计算tile列表容量，每个高斯最多占用kTilesPerScreen^2 = 64^2 = 4096个tile
    // 实际使用的时候每个高斯只添加到覆盖的tile，所以实际大小远小于这个上限
    tile_list_capacity_ = splat_count;
    tile_keys_a_.Resize(tile_list_capacity_);
    tile_keys_b_.Resize(tile_list_capacity_);
    tile_values_a_.Resize(tile_list_capacity_);
    tile_values_b_.Resize(tile_list_capacity_);

    TileListArgs tile_list_args;
    tile_list_args.keys = tile_keys_.current();
    tile_list_args.values = tile_values_.current();
    tile_list_args.capacity = tile_list_capacity_;
    SetTileListArgs(&tile_list_args);
}

void Renderer::Render(const Camera& camera, uint8_t* output_buffer, RenderStats& stats) {
    auto splat_count = positions_.Size();
    if (splat_count == 0) {
        // 测试模式，清屏显示固定颜色
        int width = camera.GetWidth();
        int height = camera.GetHeight();
        ClearScreen(reinterpret_cast<uchar4*>(output_buffer), width, height);
        return;
    }

    CudaTimer timer;

    // 预计算相机参数并上传到设备
    auto camera_data = prep_camera_data(camera);
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(g_global_args, (void*)&camera_data, sizeof(CameraData), offsetof(GlobalArgs, camera_data), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(g_global_args, &output_buffer, sizeof(uchar4*), offsetof(GlobalArgs, back_buffer), cudaMemcpyHostToDevice));

    // 步骤1：计算球面谐波颜色（视角相关）
    EvaluateSphericalHarmonics(timer, splat_count);
    stats.sh_eval_ms = timer.Elapsed();

    // 步骤2：投影高斯到屏幕空间，计算椭圆参数
    EvaluateSplatClipData(timer, splat_count);
    stats.project_ms = timer.Elapsed();

    // 步骤3：构建tile列表，每个高斯添加到它覆盖的tile
    constexpr int build_tile_num_blocks = 16;
    tile_list_size_ = BuildTileList(timer, build_tile_num_blocks, tile_list_capacity_);
    stats.build_tile_ms = timer.Elapsed();

    // 步骤4：对tile列表按深度排序，保证前到后 alpha 混合正确
    SortTileList(timer, tile_list_size_, device_temp_storage_, temp_storage_size_, tile_keys_, tile_values_);
    TileListArgs tile_list_args;
    tile_list_args.keys = tile_keys_.current();
    tile_list_args.values = tile_values_.current();
    tile_list_args.size = tile_list_size_;
    tile_list_args.capacity = tile_list_capacity_;
    SetTileListArgs(&tile_list_args);
    stats.sort_ms = timer.Elapsed();

    // 步骤5：计算每个tile的起始和结束索引
    EvaluateTileRange(timer, tile_list_size_);

    // 步骤6：光栅化所有tile，生成最终图像
    RasterizeTiles(timer);
    stats.raster_ms = timer.Elapsed();

    stats.total_ms = timer.Elapsed();
}

} // namespace gauss_render
