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

__global__ void ClearScreenKernel(uchar4* buffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        buffer[idx] = make_uchar4(0, 255, 0, 255); // RGBA格式，绿色
    }
}

void ClearScreen(uchar4* buffer, int width, int height) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    ClearScreenKernel<<<grid, block>>>(buffer, width, height);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ClearScreen kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

void RasterizeTile(CudaTimer& timer) {
    timer.Start();
    // TODO: implement kernel
    timer.Stop();
}

} // namespace gauss_render
