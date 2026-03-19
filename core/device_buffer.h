#ifndef GAUSS_RENDER_DEVICE_BUFFER_H_
#define GAUSS_RENDER_DEVICE_BUFFER_H_

#include <cuda_runtime.h>
#include <vector>

namespace gauss_render {

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t capacity) {
        Resize(capacity);
    }

    template <typename U>
    explicit DeviceBuffer(const std::vector<U>& data) {
        Resize(data.size());
        UploadFromHost(data);
    }

    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), capacity_(other.capacity_) {
        other.ptr_ = nullptr;
        other.capacity_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_ != nullptr) {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            capacity_ = other.capacity_;
            other.ptr_ = nullptr;
            other.capacity_ = 0;
        }
        return *this;
    }

    void Resize(size_t capacity) {
        if (capacity <= capacity_) {
            return;
        }
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
        capacity_ = capacity;
        cudaMalloc(&ptr_, capacity * sizeof(T));
    }

    void ResizeIfNeeded(size_t capacity) {
        Resize(capacity);
    }

    size_t Capacity() const { return capacity_; }

    T* GetDevicePtr() { return ptr_; }
    const T* GetDevicePtr() const { return ptr_; }

    void UploadFromHost(const std::vector<T>& data) {
        Resize(data.size());
        cudaMemcpy(ptr_, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice);
    }

    void DownloadToHost(std::vector<T>* data) const {
        data->resize(capacity_);
        cudaMemcpy(data->data(), ptr_, capacity_ * sizeof(T), cudaMemcpyDeviceToHost);
    }

    void Clear() {
        cudaMemset(ptr_, 0, capacity_ * sizeof(T));
    }

private:
    T* ptr_ = nullptr;
    size_t capacity_ = 0;
};

} // namespace gauss_render

#endif // GAUSS_RENDER_DEVICE_BUFFER_H_
