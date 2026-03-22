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
    // TODO: add device buffers
    // DeviceBuffer<float4> positions_;
    // DeviceBuffer<float4> scale_rotation_;
    // DeviceBuffer<float4> colors_;
    // DeviceBuffer<float4> conics_;
    // DeviceBuffer<float4> screen_ellipses_;
};

} // namespace gauss_render

#endif // GAUSS_RENDER_RENDERER_H_
