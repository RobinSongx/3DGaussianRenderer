#ifndef GAUSS_RENDER_CUDA_UTILS_H_
#define GAUSS_RENDER_CUDA_UTILS_H_

#include <cuda_runtime.h>
#include <cstdio>

namespace gauss_render {

#define CHECK_CUDA_ERROR(err) \
    do { \
        cudaError_t err_ = err; \
        if (err_ != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void Start() {
        cudaEventRecord(start_);
    }

    void Stop() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
    }

    float GetElapsedTimedMs() const {
        float ms = 0;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

} // namespace gauss_render

#endif // GAUSS_RENDER_CUDA_UTILS_H_
