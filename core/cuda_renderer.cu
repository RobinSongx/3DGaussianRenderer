#include "cuda_renderer.cuh"
#include "cuda_utils.h"

namespace gauss_render {

__constant__ GlobalArgs g_global_args;

void SetGlobalArgs(GlobalArgs* args) {
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(g_global_args, args, sizeof(GlobalArgs)));
}

void EvaluateSphericalHarmonics(CudaTimer& timer, int splat_count) {
    timer.Start();
    // TODO: implement kernel
    timer.Stop();
}

void EvaluateSplatClipData(CudaTimer& timer, int splat_count) {
    timer.Start();
    // TODO: implement kernel
    timer.Stop();
}

int BuildTileList(CudaTimer& timer, int grid_size, int tile_list_capacity) {
    timer.Start();
    // TODO: implement kernel
    timer.Stop();
    return 0;
}

void SortTileList(CudaTimer& timer, int tile_list_size,
                  void*& temp_storage, size_t& temp_storage_size) {
    timer.Start();
    // TODO: implement sorting with CUB
    timer.Stop();
}

void EvaluateTileRange(CudaTimer& timer, int tile_list_size) {
    timer.Start();
    // TODO: implement kernel
    timer.Stop();
}

void RasterizeTile(CudaTimer& timer) {
    timer.Start();
    // TODO: implement kernel
    timer.Stop();
}

} // namespace gauss_render
