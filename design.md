# 3D Gaussian Splatting Renderer 设计文档

## 项目概述

本项目是一个**从零开始用 CUDA C++ 实现的 3D Gaussian Splatting 实时渲染器**，参考 [CUDAGaussianRenderer](https://github.com/ashawkey/CUDAGaussianRenderer) 的架构设计，遵循 Google C++ 代码风格。

**目标**:
- 深入理解 3DGS 渲染原理和 CUDA 并行优化
- 实现一个可交互的实时渲染器，支持加载预训练的高斯模型
- 代码结构清晰，便于学习和扩展

**核心论文**:
- [3D Gaussian Splatting for Real-Time Radiance Field Rendering (INRIA)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [EWA Splatting](https://ieeexplore.ieee.org/document/1057888)

---

## 技术架构

### 整体流水线

```
主机端 (CPU)
├─ 初始化: GLFW窗口 + OpenGL上下文 + CUDA设备
├─ 数据加载: 解析PLY文件，读取高斯参数
├─ 数据预处理: 重排列球谐函数 → 上传到GPU显存
└─ 主循环: 处理输入 → 更新相机 → 触发CUDA渲染 → 显示结果
        ↓
GPU端 (CUDA) - 每一帧执行以下阶段:
├─ 1. evaluateSphericalHarmonics  - 计算视角依赖的RGB颜色
├─ 2. evaluateSplatClipData       - 投影高斯到屏幕，计算椭圆参数
├─ 3. buildTileList               - 构建(tile, 高斯)重叠列表
├─ 4. sortTileList                - 按tile索引和深度排序
├─ 5. evaluateTileRange           - 计算每个tile在排序后数组中的范围
└─ 6. rasterizeTile               - 分块光栅化，alpha混合输出图像
        ↓
结果通过CUDA-OpenGL互操作显示在窗口中
```

---

## 代码结构设计

**清晰模块化分离：核心算法文件夹 + 前端应用文件夹**，不做静态库封装，简单直接：

```
3DGaussianRenderer/
├── CMakeLists.txt              # CMake构建配置
├── .clang-format               # 代码格式配置（Google风格）
├── .gitignore
├── README.md                   # 项目说明，编译使用方法
├── design.md                   # 本设计文档
│
├── core/                       # 核心渲染算法（独立文件夹）
│   ├── consts.h                # 常量定义（屏幕大小，tile大小等）
│   ├── types.h                 # 基础类型定义
│   ├── camera.h                # 相机参数结构
│   ├── camera.cpp              # 相机参数计算
│   ├── camera_controls.h       # 相机交互控制（处理用户输入）
│   ├── camera_controls.cpp     # 交互实现
│   ├── gaussian_model.h         # 高斯模型数据结构
│   ├── gaussian_model.cpp       # 数据预处理（SH重排，量化）
│   ├── device_buffer.h          # CUDA设备内存RAII封装（模板声明）
│   ├── device_buffer.cu         # CUDA设备内存RAII封装（模板实现）
│   ├── cuda_error_check.h       # CUDA错误检查宏定义
│   ├── cuda_error_check.cpp     # CUDA错误检查实现
│   ├── cuda_utils.h             # CUDA工具类（计时器）
│   ├── utilities.h              # OpenGL/CUDA互操作工具类
│   ├── ply_parser.h             # PLY文件加载器
│   ├── ply_parser.cpp           # PLY加载实现
│   ├── ply_loader.h             # PLY加载器（适配层）
│   ├── ply_loader.cpp           # PLY加载实现
│   ├── renderer.h              # 渲染器接口
│   ├── renderer.cpp             # 渲染器实现
│   ├── gaussian_render.cuh      # CUDA核函数声明，数据结构
│   └── gaussian_render.cu       # 所有CUDA渲染流水线
│   ├── cuda_renderer.cuh        # CUDA渲染器接口（遗留，待整理）
│   └── cuda_renderer.cu         # CUDA渲染器实现（遗留，待整理）
│
├── app/                        # 前端应用（GLFW + OpenGL桌面）
│   ├── main.cpp                 # 主程序入口
│   ├── camera_controller.h     # 相机交互控制器
│   ├── camera_controller.cpp    # 控制器实现
│   ├── glfw_window.h           # GLFW窗口封装
│   ├── glfw_window.cpp         # 窗口实现
│   ├── opengl_display.h        # OpenGL显示输出
│   ├── opengl_display.cpp      # 显示实现
│   ├── perf_overlay.h          # 性能 overlay 显示
│   └── perf_overlay.cpp        # 性能 overlay 实现
│
├── external/                   # 第三方依赖
│   ├── glm/                    # 数学库（header-only）
│   ├── glfw/                   # 窗口管理
│   └── glew/                   # OpenGL扩展加载
│
└── data/                        # 测试数据
    └── random_cube.ply         # 随机立方体测试数据
```

### 模块职责分离

| 模块 | 职责 | 依赖 | 说明 |
|------|------|------|------|
| **core/** 核心算法 | 高斯数据结构，PLY加载，预处理，CUDA渲染流水线 | CUDA，glm，cub | 不依赖GLFW/OpenGL，纯算法 |
| **app/** 前端应用 | 窗口创建，用户输入交互，OpenGL显示输出 | core + GLFW + GLEW + OpenGL | 具体的可交互应用 |

这种设计的好处：
- 核心算法和界面分离，一目了然
- 文件夹分离就是模块分离，不需要复杂的库封装
- 以后添加其他前端直接在app下加新文件夹即可
- CMake配置简单，直接编译

---

## 抽象接口设计

### GPU缓冲区抽象接口

为了支持不同后端（CUDA/WebGPU），定义抽象基类：

```cpp
// include/backend/gpu_buffer.h
namespace gauss_render {

// GPU缓冲区抽象接口
template <typename T>
class GpuBuffer {
public:
    virtual ~GpuBuffer() = default;

    // 分配显存
    virtual void Resize(size_t count) = 0;

    // 获取当前容量
    virtual size_t Capacity() const = 0;

    // 获取GPU端原始指针（供后端核函数使用）
    virtual T* GetDevicePtr() = 0;

    // 从CPU内存上传数据到GPU
    virtual void UploadFromHost(const std::vector<T>& data) = 0;

    // 下载数据回CPU（调试用）
    virtual void DownloadToHost(std::vector<T>* data) = 0;

    // 清空内存（设置为0）
    virtual void Clear() = 0;
};

} // namespace gauss_render
```

CUDA后端只需要实现这个接口即可。

### 渲染流水线抽象基类

```cpp
// include/core/render_pipeline.h
namespace gauss_render {

class GaussianModel;
class Camera;

// 渲染流水线抽象接口
class RenderPipeline {
public:
    virtual ~RenderPipeline() = default;

    // 上传高斯数据到GPU
    virtual void UploadModel(const GaussianModel& model) = 0;

    // 执行一帧渲染，输出到颜色缓冲区
    virtual void Render(const Camera& camera, uint8_t* output_buffer) = 0;

    // 获取最新性能统计
    virtual struct RenderStats GetStats() const = 0;
};

// 性能统计
struct RenderStats {
    double sh_eval_ms;
    double project_ms;
    double build_tile_ms;
    double sort_ms;
    double raster_ms;
    double total_ms;
};

} // namespace gauss_render
```

---

## 核心数据结构设计

### CPU端数据结构（Core层，平台无关）

```cpp
// core/gaussian_model.h
namespace gauss_render {

// 单个高斯参数（CPU内存）
struct Gaussian {
    float position[3];        // 世界空间位置
    float opacity;             // 不透明度
    float scale[3];            // 三个轴的缩放因子
    float rotation[4];         // 旋转四元数 (x, y, z, w)
    float color[3];            // 静态颜色 (SH 阶数0时使用)
};

// 整个场景
class GaussianModel {
public:
    std::vector<Gaussian> gaussians;
    std::vector<float> sh_coefficients;  // 球谐系数 (degrees > 0时)
    int sh_degree = 0;                   // 球谐阶数
    int num_gaussians = 0;
    glm::vec3 bounds_min;
    glm::vec3 bounds_max;

    // 预处理：重排列球谐系数以适应GPU合并访问
    // 这一步在CPU完成，上传GPU前执行
    void PreprocessSh();

    // 获取预处理后的SH数据
    const std::vector<float>& GetAlignedSh() const { return aligned_sh_; }

private:
    std::vector<float> aligned_sh_;  // 对齐后的SH数据
};

} // namespace gauss_render
```

### 相机参数结构

```cpp
// core/camera.h
namespace gauss_render {

// 相机参数（纯数据，平台无关）
class Camera {
public:
    Camera() = default;

    // 设置视场角
    void SetFov(float fov_y_rad);

    // 设置图像分辨率
    void SetResolution(int width, int height);

    // 设置裁剪范围
    void SetNearFar(float near, float far);

    // 设置位姿（位置，朝向）
    void SetPose(const glm::vec3& position, const glm::vec3& look_at);

    // 获取ViewProjection矩阵
    glm::mat4 GetViewProjection() const;

    // 获取投影矩阵
    glm::mat4 GetProjection() const;

    // 获取视图矩阵
    glm::mat4 GetView() const;

    // 获取相机位置
    glm::vec3 GetPosition() const { return position_; }

    // 获取宽高比
    float GetAspect() const { return aspect_; }

    // 获取FOV余切
    glm::vec2 GetFovCotangent() const { return fov_cotangent_; }

    // 获取深度缩放偏移
    glm::vec2 GetDepthScaleBias() const { return depth_scale_bias_; }

    // 获取图像尺寸
    int GetWidth() const { return width_; }
    int GetHeight() const { return height_; }

private:
    glm::vec3 position_;
    glm::mat4 view_;
    glm::mat4 projection_;
    glm::mat4 view_projection_;
    glm::vec2 fov_cotangent_;
    glm::vec2 depth_scale_bias_;
    float aspect_;
    float fov_y_;
    float near_ = 0.01f;
    float far_ = 100.0f;
    int width_ = 1024;
    int height_ = 1024;
};

} // namespace gauss_render
```

### GPU端内存布局（对齐优化）

后端实现时需要遵循这个内存布局以保证性能：

```cpp
// 每个高斯占用多个float4，充分利用硬件对齐特性
// 对CUDA/WebGPU都适用
struct GpuGaussianLayout {
    float4* positions;         // xyz: 位置, w: 不透明度
    float4* scale_rotation;    // xyz: 缩放, w: 量化压缩的四元数
    float4* colors;            // rgb: 颜色, a: 不透明度（已打包）
    float4* conics;            // 逆协方差矩阵 (3个元素，float4对齐)
    float4* screen_ellipses;   // cos(angle), sin(angle), extent.x, extent.y
};

// 球谐函数内存布局优化：
// 原始布局: [gaussian][degree][channel] → 不连续，访问无法合并
// 优化后: [block][degree][channel][thread_idx] → 连续访问，完美合并
// 块大小固定为 256，一个warp内的线程访问连续地址
float* sh_coefficients;
```

**四元数量化技巧**: 四元数每个分量在 [-1, 1] 范围，映射到 [0, 1] 后每个分量用8位量化，4个分量正好32位，存入一个float的32位中，节省50%内存。

```cpp
// 编码: 4 × 8位 → 1个uint32
inline uint32_t quantize_quaternion(float q[4]) {
    uint32_t result = 0;
    for (int i = 0; i < 4; ++i) {
        uint32_t v = static_cast<uint32_t>((q[i] + 1.0f) * 0.5f * 255.0f);
        result |= (v & 0xFF) << (i * 8);
    }
    return result;
}

// 解码
inline void decode_quaternion(uint32_t packed, float q[4]) {
    for (int i = 0; i < 4; ++i) {
        uint32_t v = (packed >> (i * 8)) & 0xFF;
        q[i] = static_cast<float>(v) / 255.0f * 2.0f - 1.0f;
    }
}
```

---

## 各阶段详细设计

### 阶段 1: 球谐函数评估

**目标**: 根据当前相机视角方向，计算每个高斯的视相关颜色。

**设计要点**:
- 每个thread处理1个高斯，每个block处理256个高斯
- 利用优化后的内存布局，保证合并访问
- 使用 `__ldg` 加载只读缓存加速
- 支持动态阶数 0-4

**SH系数布局**:
```
一个block内256个高斯：
sh[0][0][0...255], sh[0][1][0...255], sh[0][2][0...255],
sh[1][0][0...255], sh[1][1][0...255], sh[1][2][0...255], ...
```

---

### 阶段 2: 裁剪空间数据评估

**目标**: 将高斯从世界空间投影到屏幕空间，计算椭圆参数和逆协方差。

**数学步骤**:
1. 高斯中心投影到裁剪空间 → 检查是否在视锥体内
2. 计算投影雅可比矩阵 J (para-perspective projection)
3. 计算 3D 协方差矩阵: Σ = R * S * S^T * R^T
4. 投影得到 2D 协方差: Σ_2D = J * Σ * J^T
5. 特征值分解得到椭圆主轴方向和长短轴
6. 计算逆协方差 conic = Σ_2D^(-1)，用于后续密度计算

**输出**:
- 屏幕空间中心坐标 (x, y)
- 深度 z
- 椭圆参数: 方向角，两个轴长
- 逆协方差矩阵 conic (3个元素)

---

### 阶段 3: 构建 Tile 列表

**这是整个算法最核心的优化阶段**。

**背景**:
- 屏幕分成 16×16 像素的 tile
- 一个 tile 内的所有计算可以放在共享内存中加速
- 需要找出每个高斯覆盖哪些 tile，建立 (tile_index, depth, splat_index) 列表

**算法设计 - 持久化线程块**:
```
grid_size = SM_count × 2  // 每个SM两个块，让CUDA调度器藏延迟
每个block持续运行直到所有高斯处理完毕:

1. 第一个warp (32线程) 加载32个高斯
   → 每个线程计算自己高斯覆盖多少个tile
   → warp内前缀扫描得到需要写入的位置

2. 整个block的所有线程参与相交测试
   → 每个线程处理一个候选tile
   → 精确测试椭圆与tile矩形是否相交

3. 所有相交结果收集到共享内存
   → 批量合并写入全局内存
   → 保证写操作合并，节省带宽

4. 如果高斯覆盖tile太多，超过共享内存容量 → 分多批处理

5. 重复直到所有高斯处理完毕 → block退出
```

**Key编码** (64位):
```
高32位: tile_index (只需要12位，因为 1024/16=64 → 64×64=4096)
低32位: depth (按深度排序，所以front-to-back处理)
```
排序后，同一个tile的所有项自然连续且按深度有序，非常巧妙。

**优化亮点**:
- 精确椭圆-矩形相交测试 → 减少无效条目，比AABB近似更优
- 持久化块减少核启动开销
- 只有第一个warp加载数据 → 节省共享内存
- 共享内存批量收集 → 全局内存合并写

---

### 阶段 4: 排序

- 使用 NVIDIA CUB 库的 `DeviceRadixSort::SortPairs`
- 只需要排序 44 位 (12位 tile + 32位 depth) → 减少排序工作量
- 第一次调用确定临时存储大小，后续复用

---

### 阶段 5: 计算 Tile 范围

排序后同一个tile的条目连续，需要记录每个tile的start和end索引：

```cpp
// 输出数组大小: 2 * num_tiles
tile_range[tile * 2] = start_index;
tile_range[tile * 2 + 1] = end_index;
```

---

### 阶段 6: 光栅化

**目标**: 对每个tile，遍历所有overlapping高斯，front-to-back alpha混合。

**设计**:
- 一个block处理一个tile (16×16 = 256像素)
- 一个thread处理一个像素
- 共享内存缓存当前chunk的32个高斯 → 减少全局内存访问
- 循环加载 → 处理 → 加载下一批

**Alpha混合**:
```cpp
float transmittance = 1.0f;
float3 color = float3{0.0f, 0.0f, 0.0f};

for (int i = start; i < end && transmittance > 1.0f / 255.0f; ++i) {
    // 计算高斯密度
    float2 diff = pixel_xy - splat_center_xy;
    float power = diff.x * diff.x * conic[0] +
                  diff.x * diff.y * conic[1] * 2.0f +
                  diff.y * diff.y * conic[2];
    power = power * 0.5f;
    float alpha = opacity * exp(-power);
    alpha = min(alpha, 0.99f);  // 防止透射率为零

    // 混合
    float weight = alpha * transmittance;
    color += weight * splat_color;
    transmittance *= (1.0f - alpha);
}

output[pixel] = color * 255.0f;
```

**早退出优化**: 当整个tile的透射率足够低 (`transmittance < 1/255`)，提前退出，节省时间。

---

## CUDA 优化策略总结

| 优化技术 | 应用位置 | 说明 |
|---------|---------|------|
| **合并访问** | 所有核函数 | SH重排，批量写tile列表 |
| **共享缓存** | buildTileList, rasterize | 减少全局内存带宽需求 |
| **常量内存** | 全局参数 | `__constant__` 低延迟访问 |
| **只读缓存** | 只读数据 | `__ldg` 指令更好利用缓存 |
| **数据压缩** | 四元数 | 8位量化，节省50%内存 |
| **持久化块** | buildTileList | 减少核启动开销 |
| **warp前缀扫描** | 负载分配 | 轻量级同步 |
| **精确相交** | tile列表构建 | 减少条目数量 |
| **位编码** | 排序key | 自然排序，无需额外整理 |
| **早退出** | 光栅化 | 透射率足够低就停止 |
| **分工协作** | buildTileList | 第一个warp专门加载 |

---

## 代码风格规范

遵循 **Google C++ Style**，通过 `.clang-format` 自动格式化：

关键规则：
- 缩进: 4空格，不使用Tab
- 最大列宽: 120字符
- 大括号换行: 每个大括号单独一行
- 指针对齐: 左对齐 (`Type* name`, 不是 `Type *name`)
- 命名:
  - 类名: 驼峰命名法 `GaussianModel`
  - 函数名: 驼峰命名法 `BuildTileList`
  - 局部变量: 小驼峰 `tileIndex`
  - 成员变量: 下划线后缀 `tile_range_`
  - 常量: k前缀 `kTileSize`
- 头文件保护: `#pragma once`
- 包含顺序: C/C++标准库 → 第三方 → 本项目内部

---

## 依赖

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++20
- OpenGL 3.3+
- GLFW (窗口管理)
- GLEW (OpenGL扩展)
- GLM (数学库)
- CUB (CUDA排序，已包含在CUDA Toolkit中)

---

## 功能清单

- [x] 设计文档
- [x] 创建目录结构
- [x] 添加.clang-format（Google风格）
- [x] 添加.gitignore
- [x] CMake构建配置
- [x] **core/ 核心算法**
  - [x] consts.h - 常量定义
  - [x] types.h - 基础类型定义
  - [x] camera.h - 相机参数结构
  - [x] camera.cpp - 相机参数计算
  - [x] camera_controls.h - 相机交互控制
  - [x] camera_controls.cpp - 交互实现
  - [x] gaussian_model.h - 高斯模型数据结构
  - [x] gaussian_model.cpp - 数据预处理（SH重排，量化）
  - [x] device_buffer.h - CUDA设备内存RAII封装
  - [x] device_buffer.cu - CUDA设备内存实现
  - [x] cuda_error_check.h - CUDA错误检查
  - [x] cuda_error_check.cpp - CUDA错误检查实现
  - [x] utilities.h - OpenGL/CUDA互操作工具
  - [x] ply_parser.h - PLY文件加载器
  - [x] ply_parser.cpp - PLY加载实现
  - [x] renderer.h - 渲染器接口
  - [x] gaussian_render.cuh - CUDA核函数声明，数据结构
  - [x] gaussian_render.cu - 所有CUDA渲染流水线
- [x] **app/ 前端应用（GLFW + OpenGL**
  - [x] main.cpp - 主程序入口
  - [x] camera_controller.h - 相机交互控制器
  - [x] camera_controller.cpp - 控制器实现
  - [x] glfw_window.h - GLFW窗口封装
  - [x] glfw_window.cpp - 窗口实现
  - [x] opengl_display.h - OpenGL显示输出
  - [x] opengl_display.cpp - 显示实现
  - [x] perf_overlay.h - 性能overlay显示
  - [x] perf_overlay.cpp - 性能overlay实现
- [x] 完整功能实现
  - [x] 球谐函数评估
  - [x] 投影和椭圆参数计算
  - [x] tile列表构建
  - [x] 排序
  - [x] 计算tile范围
  - [x] 光栅化
  - [x] CUDA-OpenGL互操作显示
  - [x] 交互控制

---

## 性能目标 (参考在RTX 3050 Laptop上)

| 高斯数量 | 目标帧率 |
|---------|---------|
| 150K | ≥ 60 FPS |
| 350K | ≥ 30 FPS |
| 800K | ≥ 20 FPS |

---

## 当前实现状态

✅ **项目已完整实现并可运行**
- 成功编译
- 成功加载PLY文件（random_cube.ply）
- 成功运行渲染
- 支持鼠标键盘相机控制
- CUDA-OpenGL互工作正常

---

## 已知局限性与未来改进

1. **分辨率固定**: 当前设计支持 1024×1024，未来可以改为动态分辨率
2. **空间加速结构**: 可以加入八叉树/网格提前剔除，进一步提升大场景性能
3. **Morton排序**: 改善光栅化阶段缓存局部性
4. **训练支持**: 当前只支持推理渲染，未来可以考虑加入训练

---

## 参考资料

1. [原论文 - 3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
2. [EWA Splatting 论文](https://ieeexplore.ieee.org/document/1057888)
3. [CUDAGaussianRenderer 参考项目](https://github.com/ashawkey/CUDAGaussianRenderer)
4. [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
