#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace gauss_render {

// CUDA设备内存缓冲区（RAII封装）
// 模板类，管理GPU显存，自动分配释放，支持动态resize
template <typename T>
class DeviceBuffer {
  private:
    T* m_Ptr{nullptr};               // 设备内存起始指针
    std::size_t m_Size{0};           // 当前使用大小（元素个数）
    std::size_t m_AllocatedSize{0};  // 已分配大小（元素个数）
    void releaseIfNeeded();          // 释放内存

  public:
    // 获取当前元素个数
    std::size_t size() const {
        return m_Size;
    };
    // 获取设备内存指针（供CUDA核函数使用）
    T* getPtr() const {
        return m_Ptr;
    };
    // 默认构造：空缓冲区
    DeviceBuffer()
        : m_Ptr(nullptr)
        , m_Size(0)
        , m_AllocatedSize(0) {};
    // 构造：分配指定大小的显存
    DeviceBuffer(std::size_t size, bool shouldClearMemory = false);
    // 构造：从CPU vector数据复制到显存
    DeviceBuffer(std::vector<T> data);
    // 析构：释放显存
    ~DeviceBuffer();
    // 如果需要，调整缓冲区大小
    // 只有当新大小大于已分配大小时才重新分配
    bool resizeIfNeeded(std::size_t size);
    // 从CPU vector复制数据到显存
    void copyFrom(const std::vector<T>& data);
    // 从显存复制数据到CPU vector
    void copyTo(std::vector<T>& data);
    // 从显存复制指定数量数据到CPU vector
    void copyTo(std::vector<T>& data, std::size_t count);
    // 用指定值清空显存（通常设为0）
    void clearMemory(int32_t value);
};

template <typename T>
void DeviceBuffer<T>::clearMemory(int32_t value) {
    if (m_Ptr != nullptr) {
        cudaMemset(m_Ptr, value, m_Size * sizeof(T));
    }
}

// 如果需要，调整缓冲区大小
// 只有当新大小大于已分配大小时才重新分配，避免频繁分配
template <typename T>
bool DeviceBuffer<T>::resizeIfNeeded(std::size_t size) {
    if (size != m_Size) {
        if (size > m_AllocatedSize) {
            releaseIfNeeded();
            m_Size = size;
            m_AllocatedSize = size;
            cudaMalloc((void**) &m_Ptr, m_AllocatedSize * sizeof(T));
            return true;
        }
        m_Size = size;
    }
    return false;
}

// 释放显存（如果存在）
template <typename T>
void DeviceBuffer<T>::releaseIfNeeded() {
    if (m_Ptr != nullptr) {
        cudaFree(m_Ptr);
        m_Ptr = nullptr;
    }
    m_Size = 0;
    m_AllocatedSize = 0;
}

// 构造：分配指定大小的显存
template <typename T>
DeviceBuffer<T>::DeviceBuffer(std::size_t size, bool shouldClearMemory) {
    resizeIfNeeded(size);
    if (shouldClearMemory) {
        clearMemory(0);
    }
};

// 构造：从CPU vector数据复制到显存
template <typename T>
DeviceBuffer<T>::DeviceBuffer(std::vector<T> data) {
    copyFrom(data);
};

// 析构：释放显存
template <typename T>
DeviceBuffer<T>::~DeviceBuffer() {
    releaseIfNeeded();
};

// 从CPU vector复制数据到显存
template <typename T>
void DeviceBuffer<T>::copyFrom(const std::vector<T>& data) {
    resizeIfNeeded(data.size());
    if (data.size() != 0) {
        cudaMemcpy(m_Ptr, &data[0], data.size() * sizeof(T), cudaMemcpyHostToDevice);
    }
}

// 从显存复制指定数量数据到CPU vector
template <typename T>
void DeviceBuffer<T>::copyTo(std::vector<T>& data, std::size_t count) {
    if (count <= m_Size) {
        if (data.size() != count) {
            data.resize(count);
        }
        cudaMemcpy(&data[0], m_Ptr, count * sizeof(T), cudaMemcpyDeviceToHost);
    }
}

// 从显存复制完整数据到CPU vector
template <typename T>
void DeviceBuffer<T>::copyTo(std::vector<T>& data) {
    copyTo(data, data.size());
}

} // namespace gauss_render
