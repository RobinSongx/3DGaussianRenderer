#ifndef GAUSS_RENDER_CUDA_RENDERER_CUH_
#define GAUSS_RENDER_CUDA_RENDERER_CUH_

#include "types.h"
#include "camera.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>

namespace gauss_render {

struct CameraData {
    glm::vec3 position;
    float aspect;
    glm::mat4 projection;
    glm::mat4 view;
    glm::mat4 view_projection;
    glm::vec2 fov_cotangent;
    glm::vec2 depth_scale_bias;
};

struct GlobalArgs {
    int splat_count;
    int sh_degree;
    int sh_count;
    float4* positions;
    float4* scale_rotation;
    float4* colors;
    float4* conics;
    float4* screen_ellipses;
    float2* position_xy_clip;
    float* position_z_clip;
    float* spherical_harmonics;
    int2* tile_range;
    uchar4* back_buffer;
    CameraData camera_data;
};

void SetGlobalArgs(GlobalArgs* args);

void EvaluateSphericalHarmonics(CudaTimer& timer, int splat_count);
void EvaluateSplatClipData(CudaTimer& timer, int splat_count);
int BuildTileList(CudaTimer& timer, int grid_size, int tile_list_capacity);
void SortTileList(CudaTimer& timer, int tile_list_size,
                  void*& temp_storage, size_t& temp_storage_size);
void EvaluateTileRange(CudaTimer& timer, int tile_list_size);
void RasterizeTile(CudaTimer& timer);

} // namespace gauss_render

#endif // GAUSS_RENDER_CUDA_RENDERER_CUH_
