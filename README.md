# 3DGaussianRenderer
CUDA加速的3D高斯溅射实时渲染器

## 依赖

- CMake 3.18+
- CUDA Toolkit 11.0+
- Visual Studio 2022 (Windows必须，CUDA不支持MinGW)
- OpenGL

## 编译步骤 (Windows)

### 1. 环境准备
- 安装 [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- 安装 Visual Studio 2022，勾选 "C++开发工具" 和 "MSVC编译器"
- 安装 CMake

### 2. 配置和编译

打开 **x86-64 Native Tools Command Prompt for VS 2022** 或者在PowerShell中运行：

```cmd
cd 3DGaussianRenderer
mkdir -p build
cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```

### 3. 复制依赖DLL

编译完成后，需要将GLEW的DLL复制到可执行文件目录：

```cmd
copy ..\external\glew\bin\Release\x64\glew32.dll .\bin\Release\
```

### 4. 修复说明

如果遇到编译错误，请检查：

1. **CMAKE_CUDA_FLAGS 已经添加了 `-Xcompiler=/utf-8` 解决中文编码问题**
2. **编译选项需要使用生成器表达式只作用于C++，避免nvcc接收MSVC选项 `/W4`** - 已在CMakeLists.txt修复
3. **`cuda_renderer.cuh` 需要包含 `cuda_utils.h` 并且 `CameraData` 需要定义在 `GlobalArgs` 之前** - 已修复

## 运行

需要提供一个3D高斯溅射训练输出的PLY文件：

```cmd
cd bin\Release
gaussian_render.exe path\to\your\point_cloud.ply
```

### controls

- **鼠标拖动** - 旋转相机
- **滚轮** - 缩放
- **WASD** - 移动相机
- **ESC** - 退出

## 项目结构

```
├── app/             # 应用层（GLFW窗口、OpenGL显示、相机控制）
├── core/           # 核心渲染（CUDA核函数、高斯模型、PLY加载）
└── external/       # 第三方库（GLFW、GLEW、GLM）
```

## 性能统计

**实时叠加显示**：程序会在窗口左上角实时显示性能数据：
- FPS - 当前帧率
- Gaussians - 高斯数量
- Total - 总渲染时间
- SH - 球谐函数评估时间
- Project - 投影时间
- Build - 构建tile列表时间
- Sort - 排序时间  
- Raster - 光栅化时间

**退出统计**：程序退出后会在控制台输出累计平均耗时：
```
=== Average timings per frame ===
SH eval:     X.X ms
Projection:  X.X ms
Build tile:  X.X ms
Sort:        X.X ms
Rasterize:   X.X ms
Total:       X.X ms
FPS:         XX.X
```

