#ifndef GAUSS_RENDER_CUDA_UTILS_H_
#define GAUSS_RENDER_CUDA_UTILS_H_

#include <cuda_runtime.h>
#include <cstdio>

namespace gauss_render {

// CUDA错误检查宏
// 如果CUDA调用失败，打印错误信息并退出程序
#define CHECK_CUDA_ERROR(err) \
    do { \
        cudaError_t err_ = err; \
        if (err_ != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// CUDA计时器
// 使用cudaEvent测量CUDA核函数执行时间
class CudaTimer {
public:
    // 构造：创建开始和停止事件
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    // 析构：销毁事件
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    // 开始计时
    void Start() {
        cudaEventRecord(start_);
    }

    // 停止计时
    void Stop() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
    }

    // 获取经过的时间（毫秒）
    float GetElapsedTimedMs() const {
        float ms = 0;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    cudaEvent_t start_;  // 开始事件
    cudaEvent_t stop_;   // 停止事件
};

} // namespace gauss_render

#endif // GAUSS_RENDER_CUDA_UTILS_H_
