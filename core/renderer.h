#ifndef GAUSS_RENDER_RENDERER_H_
#define GAUSS_RENDER_RENDERER_H_

#include "types.h"
#include "camera.h"
#include "gaussian_model.h"
#include "device_buffer.h"

namespace gauss_render {

class Renderer {
public:
    Renderer() = default;
    ~Renderer() = default;

    void UploadModel(const GaussianModel& model);
    void Render(const Camera& camera, uint8_t* output_buffer, RenderStats& stats);

private:
    // GPU缓冲区
    DeviceBuffer<float4> positions_;
    DeviceBuffer<float4> scale_rotation_;
    DeviceBuffer<float4> colors_;
    DeviceBuffer<float4> conics_;
    DeviceBuffer<float4> screen_ellipses_;
    DeviceBuffer<float2> position_xy_clip_;
    DeviceBuffer<float> position_z_clip_;
    DeviceBuffer<float> sh_;
    DeviceBuffer<int2> tile_ranges_{kTotalTiles * 2};

    // tile列表和排序用的双缓冲
    int tile_list_capacity_{0};
    int tile_list_size_{0};
    DeviceBuffer<uint64_t> tile_keys_a_;
    DeviceBuffer<uint64_t> tile_keys_b_;
    DeviceBuffer<int32_t> tile_values_a_;
    DeviceBuffer<int32_t> tile_values_b_;
    DoubleBuffer<uint64_t> tile_keys_{tile_keys_a_.Ptr(), tile_keys_b_.Ptr()};
    DoubleBuffer<int32_t> tile_values_{tile_values_a_.Ptr(), tile_values_b_.Ptr()};

    // cub排序临时存储
    void* device_temp_storage_{nullptr};
    size_t temp_storage_size_{0};
};

} // namespace gauss_render

#endif // GAUSS_RENDER_RENDERER_H_
