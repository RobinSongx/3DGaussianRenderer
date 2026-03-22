#ifndef GAUSS_RENDER_RENDERER_H_
#define GAUSS_RENDER_RENDERER_H_

#include "types.h"
#include "camera.h"
#include "gaussian_model.h"
#include "device_buffer.h"

namespace gauss_render {

// 3D高斯渲染器
// 管理所有GPU缓冲区，执行完整的渲染流水线
class Renderer {
public:
    Renderer() = default;
    ~Renderer() = default;

    // 将高斯模型上传到GPU显存
    void UploadModel(const GaussianModel& model);
    // 执行一帧渲染，输出颜色到output_buffer
    void Render(const Camera& camera, uint8_t* output_buffer, RenderStats& stats);

private:
    // 高斯参数GPU缓冲区
    DeviceBuffer<float4> positions_;         // xyz: 位置, w: 不透明度
    DeviceBuffer<float4> scale_rotation_;    // xyz: 缩放, w: 量化旋转四元数
    DeviceBuffer<float4> colors_;           // rgb: 颜色, a: 不透明度（激活后）
    DeviceBuffer<float4> conics_;           // 逆协方差矩阵（3个元素，float4对齐）
    DeviceBuffer<float4> screen_ellipses_;  // 屏幕空间椭圆参数 (cos, sin, extent_x, extent_y)
    DeviceBuffer<float2> position_xy_clip_; // 裁剪空间XY坐标
    DeviceBuffer<float> position_z_clip_;   // 裁剪空间Z坐标（深度）
    DeviceBuffer<float> sh_;                 // 对齐后的球谐系数
    DeviceBuffer<int2> tile_ranges_;        // 每个tile在排序数组中的[start, end]范围

    // tile列表和排序用的双缓冲
    // 每次排序后交换前后缓冲，避免重复分配
    int tile_list_capacity_{0};             // tile列表容量
    int tile_list_size_{0};                 // 当前tile列表大小
    DeviceBuffer<uint64_t> tile_keys_a_;    // 排序key缓冲区A
    DeviceBuffer<uint64_t> tile_keys_b_;    // 排序key缓冲区B
    DeviceBuffer<int32_t> tile_values_a_;  // 排序value缓冲区A
    DeviceBuffer<int32_t> tile_values_b_;  // 排序value缓冲区B
    DoubleBuffer<uint64_t> tile_keys_{tile_keys_a_.getPtr(), tile_keys_b_.getPtr()};
    DoubleBuffer<int32_t> tile_values_{tile_values_a_.getPtr(), tile_values_b_.getPtr()};

    // CUB排序需要的临时存储
    // 第一次排序时确定大小，后续复用
    void* device_temp_storage_{nullptr};
    size_t temp_storage_size_{0};
};

} // namespace gauss_render

#endif // GAUSS_RENDER_RENDERER_H_
