#ifndef GAUSS_RENDER_PERF_OVERLAY_H_
#define GAUSS_RENDER_PERF_OVERLAY_H_

#include "../core/types.h"
#include <cstdint>

namespace gauss_render {

void DrawPerformanceOverlay(
    uint8_t* buffer,
    int width,
    int height,
    int num_gaussians,
    const RenderStats& stats,
    double fps);

} // namespace gauss_render

#endif // GAUSS_RENDER_PERF_OVERLAY_H_
