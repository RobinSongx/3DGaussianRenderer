#include "cuda_renderer.cuh"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>

namespace gauss_render {

// 设备常量，存储全局参数
__constant__ GlobalArgs g_global_args;
__constant__ TileListArgs g_tile_list_args;

// 全局计数器，用于构建tile列表时统计
__device__ __managed__ int32_t g_splat_counter;
__device__ __managed__ int32_t g_tile_counter;

// 从设备内存加载常量（使用只读缓存加速访问）
template <typename T>
__device__ __forceinline__ T load_ro(T* ptr) {
    return __ldg(reinterpret_cast<const T*>(ptr));
}

// CUDA内置float2转glm::vec2
__device__ __host__ __forceinline__ glm::vec2 float2_to_glm(const float2 v) {
    return glm::vec2(v.x, v.y);
}

// CUDA内置float4转glm::vec4
__device__ __host__ __forceinline__ glm::vec4 float4_to_glm(const float4 v) {
    return glm::vec4(v.x, v.y, v.z, v.w);
}

// CUDA内置int4转glm::ivec4
__device__ __host__ __forceinline__ glm::ivec4 int4_to_glm(const int4 v) {
    return glm::ivec4(v.x, v.y, v.z, v.w);
}

// 解码压缩的四元数
__device__ glm::vec4 decode_quat(uint32_t v) {
    return glm::vec4((v >> 24u) & 0xFF, (v >> 16u) & 0xFF, (v >> 8u) & 0xFF, v & 0xFF) / 255.0f;
}

// 球面谐波计算，计算视角相关的RGB颜色
// 代码由sh_gen.py生成，支持最高4阶SH
// clang-format off
__device__ glm::vec3 spherical_harmonics(const int degree, const glm::vec3& dir, const float* sh, const int stride) {
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    glm::vec3 result = glm::vec3(0.0f);

    // 0阶
    float r0 = load_ro(&sh[0 * stride]);
    float g0 = load_ro(&sh[1 * stride]);
    float b0 = load_ro(&sh[2 * stride]);
    glm::vec3 sh0 = glm::vec3(r0, g0, b0);
    result += (0.282094792F) * sh0;

    // 1阶
    if (degree > 0) {
        float r1 = load_ro(&sh[3 * stride]);
        float g1 = load_ro(&sh[4 * stride]);
        float b1 = load_ro(&sh[5 * stride]);
        glm::vec3 sh1 = glm::vec3(r1, g1, b1);
        float r2 = load_ro(&sh[6 * stride]);
        float g2 = load_ro(&sh[7 * stride]);
        float b2 = load_ro(&sh[8 * stride]);
        glm::vec3 sh2 = glm::vec3(r2, g2, b2);
        float r3 = load_ro(&sh[9 * stride]);
        float g3 = load_ro(&sh[10 * stride]);
        float b3 = load_ro(&sh[11 * stride]);
        glm::vec3 sh3 = glm::vec3(r3, g3, b3);

        result +=
            (0.488602512F * y) * sh1 +
            (0.488602512F * z) * sh2 +
            (0.488602512F * x) * sh3;

        // 2阶
        if (degree > 1) {
            float xx = x * x;
            float yy = y * y;
            float zz = z * z;

            float r4 = load_ro(&sh[12 * stride]);
            float g4 = load_ro(&sh[13 * stride]);
            float b4 = load_ro(&sh[14 * stride]);
            glm::vec3 sh4 = glm::vec3(r4, g4, b4);
            float r5 = load_ro(&sh[15 * stride]);
            float g5 = load_ro(&sh[16 * stride]);
            float b5 = load_ro(&sh[17 * stride]);
            glm::vec3 sh5 = glm::vec3(r5, g5, b5);
            float r6 = load_ro(&sh[18 * stride]);
            float g6 = load_ro(&sh[19 * stride]);
            float b6 = load_ro(&sh[20 * stride]);
            glm::vec3 sh6 = glm::vec3(r6, g6, b6);
            float r7 = load_ro(&sh[21 * stride]);
            float g7 = load_ro(&sh[22 * stride]);
            float b7 = load_ro(&sh[23 * stride]);
            glm::vec3 sh7 = glm::vec3(r7, g7, b7);
            float r8 = load_ro(&sh[24 * stride]);
            float g8 = load_ro(&sh[25 * stride]);
            float b8 = load_ro(&sh[26 * stride]);
            glm::vec3 sh8 = glm::vec3(r8, g8, b8);

            result +=
                (1.09254843F * x * y) * sh4 +
                (1.09254843F * y * z) * sh5 +
                (-0.946174696F * xx - 0.946174696F * yy + 0.630783131F) * sh6 +
                (1.09254843F * x * z) * sh7 +
                (0.546274215F * (x - y) * (x + y)) * sh8;

            // 3阶
            if (degree > 2) {
                float r9  = load_ro(&sh[27 * stride]);
                float g9  = load_ro(&sh[28 * stride]);
                float b9  = load_ro(&sh[29 * stride]);
                glm::vec3 sh9  = glm::vec3(r9, g9, b9);
                float r10 = load_ro(&sh[30 * stride]);
                float g10 = load_ro(&sh[31 * stride]);
                float b10 = load_ro(&sh[32 * stride]);
                glm::vec3 sh10 = glm::vec3(r10, g10, b10);
                float r11 = load_ro(&sh[33 * stride]);
                float g11 = load_ro(&sh[34 * stride]);
                float b11 = load_ro(&sh[35 * stride]);
                glm::vec3 sh11 = glm::vec3(r11, g11, b11);
                float r12 = load_ro(&sh[36 * stride]);
                float g12 = load_ro(&sh[37 * stride]);
                float b12 = load_ro(&sh[38 * stride]);
                glm::vec3 sh12 = glm::vec3(r12, g12, b12);
                float r13 = load_ro(&sh[39 * stride]);
                float g13 = load_ro(&sh[40 * stride]);
                float b13 = load_ro(&sh[41 * stride]);
                glm::vec3 sh13 = glm::vec3(r13, g13, b13);
                float r14 = load_ro(&sh[42 * stride]);
                float g14 = load_ro(&sh[43 * stride]);
                float b14 = load_ro(&sh[44 * stride]);
                glm::vec3 sh14 = glm::vec3(r14, g14, b14);
                float r15 = load_ro(&sh[45 * stride]);
                float g15 = load_ro(&sh[46 * stride]);
                float b15 = load_ro(&sh[47 * stride]);
                glm::vec3 sh15 = glm::vec3(r15, g15, b15);

                result +=
                    (0.295021795F * y * (6.0F * xx - 2.0F * yy)) * sh9 +
                    (2.89061144F * x * y * z) * sh10 +
                    (3.6563664F * y * (-0.625F * xx - 0.625F * yy + 0.5F)) * sh11 +
                    (0.373176333F * z * (-5.0F * xx - 5.0F * yy + 2.0F)) * sh12 +
                    (0.457045799F * x * (-5.0F * xx - 5.0F * yy + 4.0F)) * sh13 +
                    (1.44530572F * z * (x - y) * (x + y)) * sh14 +
                    (0.59004359F * x * (xx - 3.0F * yy)) * sh15;

                // 4阶
                if (degree > 3) {
                    float r16 = load_ro(&sh[48 * stride]);
                    float g16 = load_ro(&sh[49 * stride]);
                    float b16 = load_ro(&sh[50 * stride]);
                    glm::vec3 sh16 = glm::vec3(r16, g16, b16);
                    float r17 = load_ro(&sh[51 * stride]);
                    float g17 = load_ro(&sh[52 * stride]);
                    float b17 = load_ro(&sh[53 * stride]);
                    glm::vec3 sh17 = glm::vec3(r17, g17, b17);
                    float r18 = load_ro(&sh[54 * stride]);
                    float g18 = load_ro(&sh[55 * stride]);
                    float b18 = load_ro(&sh[56 * stride]);
                    glm::vec3 sh18 = glm::vec3(r18, g18, b18);
                    float r19 = load_ro(&sh[57 * stride]);
                    float g19 = load_ro(&sh[58 * stride]);
                    float b19 = load_ro(&sh[59 * stride]);
                    glm::vec3 sh19 = glm::vec3(r19, g19, b19);
                    float r20 = load_ro(&sh[60 * stride]);
                    float g20 = load_ro(&sh[61 * stride]);
                    float b20 = load_ro(&sh[62 * stride]);
                    glm::vec3 sh20 = glm::vec3(r20, g20, b20);
                    float r21 = load_ro(&sh[63 * stride]);
                    float g21 = load_ro(&sh[64 * stride]);
                    float b21 = load_ro(&sh[65 * stride]);
                    glm::vec3 sh21 = glm::vec3(r21, g21, b21);
                    float r22 = load_ro(&sh[66 * stride]);
                    float g22 = load_ro(&sh[67 * stride]);
                    float b22 = load_ro(&sh[68 * stride]);
                    glm::vec3 sh22 = glm::vec3(r22, g22, b22);
                    float r23 = load_ro(&sh[69 * stride]);
                    float g23 = load_ro(&sh[70 * stride]);
                    float b23 = load_ro(&sh[71 * stride]);
                    glm::vec3 sh23 = glm::vec3(r23, g23, b23);
                    float r24 = load_ro(&sh[72 * stride]);
                    float g24 = load_ro(&sh[73 * stride]);
                    float b24 = load_ro(&sh[74 * stride]);
                    glm::vec3 sh24 = glm::vec3(r24, g24, b24);

                    result +=
                        (2.50334294F * x * y * (xx - yy)) * sh16 +
                        (0.295021795F * y * z * (18.0F * xx - 6.0F * yy)) * sh17 +
                        (1.26156626F * x * y * (-5.25F * xx - 5.25F * yy + 4.5F)) * sh18 +
                        (1.78412412F * y * z * (-2.625F * xx - 2.625F * yy + 1.5F)) * sh19 +
                        (7.40498828F * xx * yy - 4.23142188F * xx + 3.70249414F * xx * xx - 4.23142188F * yy + 3.70249414F * yy * yy + 0.846284375F) * sh20 +
                        (0.669046544F * x * z * (-7.0F * xx - 7.0F * yy + 4.0F)) * sh21 +
                        (-0.473087348F * (x - y) * (x + y) * (7.0F * xx + 7.0F * yy - 6.0F)) * sh22 +
                        (1.77013077F * x * z * (xx - 3.0F * yy)) * sh23 +
                        (-3.75501441F * xx * yy + 0.625835735F * xx * xx + 0.625835735F * yy * yy) * sh24;
                }
            }
        }
    }
    result += 0.5f;
    result.x = min(max(result.x, 0.0f), 1.0f);
    result.y = min(max(result.y, 0.0f), 1.0f);
    result.z = min(max(result.z, 0.0f), 1.0f);
    return result;
}
// clang-format on

// 球面谐波计算内核：每个高斯计算视角相关的RGB颜色
__global__ void evaluate_spherical_harmonics_kernel() {
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < g_global_args.splat_count) {
        // 读取位置信息，位置w分量存储透明度
        auto position_data = load_ro(&g_global_args.positions[index]);
        auto splat_position = glm::vec3(float4_to_glm(position_data));
        auto opacity = position_data.w;

        // 计算从相机中心到高斯中心的单位方向向量
        auto ray_dir = glm::normalize(g_global_args.camera_data.position - splat_position);

        // 根据SH度数计算颜色，每个高斯占 (degree+1)^2 个SH系数
        auto sh_idx = index * (g_global_args.sh_degree + 1) * (g_global_args.sh_degree + 1);
        auto sh_color = spherical_harmonics(g_global_args.sh_degree, ray_dir, &g_global_args.spherical_harmonics[sh_idx], 
                                         (g_global_args.sh_degree + 1) * (g_global_args.sh_degree + 1));

        // 保存结果：颜色 + 透明度
        g_global_args.colors[index] = float4{sh_color.r, sh_color.g, sh_color.b, opacity};
    }
}

// 计算球面谐波颜色（入口函数）
void EvaluateSphericalHarmonics(CudaTimer& timer, int splat_count) {
    constexpr int block_size = 256;
    const int num_blocks = (splat_count + block_size - 1) / block_size;
    const dim3 block_dim = dim3(block_size);
    const dim3 grid_dim = dim3(num_blocks);

    timer.Start();
    evaluate_spherical_harmonics_kernel<<<grid_dim, block_dim>>>();
    timer.Stop();

    if (cudaGetLastError() != cudaSuccess) {
        printf("kernel error evaluate_spherical_harmonics\n");
    }
}

// 矩形结构
struct Rect {
    __device__ __host__ Rect() : min(0.0f), max(0.0f) {}
    glm::vec2 min;
    glm::vec2 max;

    __device__ __host__ glm::vec2 getCenter() const {
        return min + (max - min) * 0.5f;
    }
};

// 椭圆结构
struct Ellipse {
    __device__ __host__ Ellipse() : center(0.0f), extent(0.0f), cos_sin(0.0f) {}
    glm::vec2 center;
    glm::vec2 extent;
    glm::vec2 cos_sin;

    __device__ __host__ glm::vec2 getPrincipalAxis() const {
        return cos_sin * extent.x;
    }
    __device__ __host__ glm::vec2 getMinorAxis() const {
        return glm::vec2(cos_sin.y, -cos_sin.x) * extent.y;
    }
};

// 点转换到椭圆局部坐标
__device__ __forceinline__ glm::vec2 convert_to_ellipse_coords(const Ellipse& ellipse, const glm::vec2& point) {
    auto delta = point - ellipse.center;
    return glm::vec2((delta.x * ellipse.cos_sin.x + delta.y * ellipse.cos_sin.y) / ellipse.extent.x,
                     (delta.y * ellipse.cos_sin.x - delta.x * ellipse.cos_sin.y) / ellipse.extent.y);
}

// 检查线段与单位圆是否相交
__device__ __forceinline__ bool intersects_unit_circle(const glm::vec2& v0, const glm::vec2& v1) {
    auto delta = v1 - v0;
    auto length_sqr = glm::length2(delta);
    auto t = __saturatef(glm::dot(-v0, delta) / length_sqr);
    auto projection = v0 + t * delta;
    return glm::length2(projection) < 1.0f;
}

// 检查椭圆和矩形是否重叠
__device__ __forceinline__ bool ellipse_rect_overlap(const Ellipse& ellipse, const Rect& rect) {
    // 椭圆中心在矩形内直接返回真
    auto overlaps = 
        ellipse.center.x > rect.min.x && 
        ellipse.center.x < rect.max.x && 
        ellipse.center.y > rect.min.y && 
        ellipse.center.y < rect.max.y;

    // 矩形中心在椭圆内直接返回真
    overlaps |= glm::length2(convert_to_ellipse_coords(ellipse, rect.getCenter())) < 1.0f;

    // 检查矩形四条边是否与椭圆相交
    glm::vec2 points[4];
    points[0] = convert_to_ellipse_coords(ellipse, rect.min);
    points[1] = convert_to_ellipse_coords(ellipse, glm::vec2(rect.max.x, rect.min.y));
    points[2] = convert_to_ellipse_coords(ellipse, rect.max);
    points[3] = convert_to_ellipse_coords(ellipse, glm::vec2(rect.min.x, rect.max.y));

    overlaps |= intersects_unit_circle(points[0], points[1]);
    overlaps |= intersects_unit_circle(points[1], points[2]);
    overlaps |= intersects_unit_circle(points[2], points[3]);
    overlaps |= intersects_unit_circle(points[3], points[0]);

    return overlaps;
}

// 计算椭圆的AABB包围盒
__device__ __forceinline__ Rect get_aabb_rect(const Ellipse& ellipse) {
    auto right = ellipse.getPrincipalAxis();
    auto up = ellipse.getMinorAxis();

    Rect rect;
    rect.min = glm::vec2(1e12f, 1e12f);
    rect.max = glm::vec2(-1e12f, -1e12f);

#pragma unroll
    for (int j = 0; j != 4; ++j) {
        auto bottom_bit = j & 1;
        auto top_bit = j >> 1;
        auto r = (bottom_bit ^ top_bit) * 2.0f - 1.0f;
        auto u = top_bit * 2.0f - 1.0f;
        auto v = right * r + up * u;
        rect.min = glm::min(rect.min, v);
        rect.max = glm::max(rect.max, v);
    }

    rect.min += ellipse.center;
    rect.max += ellipse.center;
    return rect;
}

// 生成排序key：tile索引高32位，深度低32位
__device__ __forceinline__ uint64_t get_sort_key(int32_t tile_index, float clip_depth) {
    // 深度范围 [-1, 1]，量化到32位
    auto quantized_depth = (uint32_t)(glm::clamp((clip_depth + 1.0f) * 0.5f, 0.0f, 1.0f) * 0xFFFFFFFFU);
    return ((uint64_t)tile_index << 32) | quantized_depth;
}

// 高斯裁剪空间数据计算内核：将世界空间高斯投影到屏幕空间，计算椭圆参数
__global__ void evaluate_splat_clip_data_kernel() {
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < g_global_args.splat_count) {
        // 读取位置和缩放旋转数据
        auto position_data = load_ro(&g_global_args.positions[index]);
        auto scale_rotation_data = load_ro(&g_global_args.scale_rotation[index]);

        auto position = glm::vec3(float4_to_glm(position_data));

        // 分解缩放和旋转，构建协方差矩阵
        glm::mat3 scale_mat = glm::mat3(0);
        scale_mat[0][0] = scale_rotation_data.x;
        scale_mat[1][1] = scale_rotation_data.y;
        scale_mat[2][2] = scale_rotation_data.z;

        // 解码压缩的四元数
        auto rotation_values = decode_quat(reinterpret_cast<uint32_t&>(scale_rotation_data.w)) * 2.0f - 1.0f;
        auto rotation = glm::quat(rotation_values.w, rotation_values.x, rotation_values.y, rotation_values.z);

        // 计算世界空间协方差：RS * (RS)^T = R S S R^T
        auto RS = glm::mat3_cast(rotation) * scale_mat;
        auto cov_world = RS * glm::transpose(RS);

        // 计算视图空间中的高斯中心
        auto view_position = (glm::vec3)(g_global_args.camera_data.view * glm::vec4(position, 1));

        // 获取相机预计算的参数
        auto fov_cotangent = g_global_args.camera_data.fov_cotangent;
        auto depth_scale_bias = g_global_args.camera_data.depth_scale_bias;

        // 仿射投影，准平行近似（论文中的公式）
        // 将世界空间协方差投影到屏幕空间
        auto z_rcp = 1.0f / view_position.z;
        auto z_rcp_sqr = z_rcp * z_rcp;
        auto scale_x = -fov_cotangent.x * z_rcp;
        auto scale_y = -fov_cotangent.y * z_rcp;
        auto fov_cotangent_trans = fov_cotangent * glm::vec2(view_position.x, view_position.y);
        auto shear_x = fov_cotangent_trans.x * z_rcp_sqr;
        auto shear_y = fov_cotangent_trans.y * z_rcp_sqr;
        auto trans_x = -fov_cotangent_trans.x * z_rcp;
        auto trans_y = -fov_cotangent_trans.y * z_rcp;

        // 构建仿射投影矩阵
        glm::mat4 affine_proj = glm::mat4(1);
        affine_proj[0][0] = scale_x;
        affine_proj[1][1] = scale_y;
        affine_proj[2][2] = depth_scale_bias.x;
        affine_proj[2][0] = shear_x;
        affine_proj[2][1] = shear_y;
        affine_proj[3][0] = trans_x;
        affine_proj[3][1] = trans_y;
        affine_proj[3][2] = depth_scale_bias.y;

        // 计算裁剪空间协方差
        auto view_proj = (glm::mat3)affine_proj * (glm::mat3)g_global_args.camera_data.view;
        auto cov_clip = view_proj * glm::transpose(cov_world) * glm::transpose(view_proj);
        auto clip_position = (glm::vec3)(affine_proj * glm::vec4(view_position, 1));

        // 添加极小值确保每个高斯至少覆盖一个像素
        constexpr float kPi = 3.14159265359f;
        constexpr float kTexelSizeClip = 2.0f / (float)kScreenSize;
        constexpr float kTraceBump = (1.0f / kPi) * kTexelSizeClip * kTexelSizeClip;
        cov_clip[0][0] += kTraceBump;
        cov_clip[1][1] += kTraceBump;

        // 对2D协方差进行特征分解，得到椭圆参数
        auto det = cov_clip[0][0] * cov_clip[1][1] - cov_clip[1][0] * cov_clip[1][0];
        auto mid = 0.5f * (cov_clip[0][0] + cov_clip[1][1]);
        constexpr float epsilon = 1e-12f;
        auto radius = glm::sqrt(glm::max(epsilon, mid * mid - det));
        auto lambda0 = mid + radius;
        auto lambda1 = glm::max(0.0f, mid - radius);

        // 计算特征向量
        auto eigen_vector0 = glm::normalize(glm::vec2(cov_clip[1][0], lambda0 - cov_clip[0][0]));
        auto eigen_vector1 = glm::normalize(glm::vec2(eigen_vector0.y / g_global_args.camera_data.aspect, -eigen_vector0.x));

        // 计算角度和半径（3sigma规则）
        auto angle = glm::atan(eigen_vector0.y, eigen_vector0.x);
        auto extent = glm::sqrt(glm::vec2(lambda0, lambda1)) * 3.0f;

        // 计算逆协方差（圆锥矩阵），因为矩阵对称，只需要存储三个元素
        auto inv_det = 1.0f / glm::max(epsilon, det);
        auto conic = glm::vec3(cov_clip[1][1], -cov_clip[1][0], cov_clip[0][0]) * inv_det;

        // 视锥体裁剪：检查椭圆是否可见
        auto edge = glm::step(glm::vec3(-1.0f), clip_position) * glm::step(clip_position, glm::vec3(1.0f));
        auto is_visible = edge.x * edge.y * edge.z * glm::step(0.0f, lambda1);

        // 不可见的高斯设置为非法值，后面会被过滤掉
        clip_position = glm::mix(glm::vec3(-128.0f), clip_position, is_visible);
        extent *= is_visible;

        // 保存结果到全局内存
        g_global_args.position_xy_clip[index] = float2{clip_position.x, clip_position.y};
        g_global_args.position_z_clip[index] = clip_position.z;
        g_global_args.screen_ellipses[index] = float4{glm::cos(angle), glm::sin(angle), extent.x, extent.y};
        g_global_args.conics[index] = float4{conic.x, conic.y, conic.z, 0.0f};
    }
}

// 投影高斯到裁剪空间（入口函数）
void EvaluateSplatClipData(CudaTimer& timer, int splat_count) {
    constexpr int block_size = 256;
    const int num_blocks = (splat_count + block_size - 1) / block_size;
    const dim3 block_dim = dim3(block_size);
    const dim3 grid_dim = dim3(num_blocks);

    timer.Start();
    evaluate_splat_clip_data_kernel<<<grid_dim, block_dim>>>();
    timer.Stop();

    if (cudaGetLastError() != cudaSuccess) {
        printf("kernel error evaluate_splat_clip_data\n");
    }
}

// 常量定义
constexpr uint32_t kWarpMask = 0xffffffff;
constexpr int32_t kWarpSize = 32;
constexpr int32_t kWarpHalfSize = kWarpSize / 2;
constexpr int32_t kBuildTileListsThreadsPerGroup = kWarpSize * 8;
constexpr uint32_t kMaxUint32 = 0xFFFFFFFF;

// 构建tile列表内核：将每个高斯添加到它覆盖的所有tile中
__global__ void build_tile_list_kernel() {
    // 共享内存：每个线程块处理一块高斯，通过共享内存协作
    __shared__ int32_t s_splat_start_index;
    __shared__ int32_t s_splat_count;
    __shared__ int4 s_tiles_rects[kWarpSize];
    __shared__ float4 s_screen_ellipse[kWarpSize];
    __shared__ float2 s_position_xy_clip[kWarpSize];
    __shared__ float s_depths[kWarpSize];
    __shared__ int s_tile_exclusive_scan[kWarpSize];
    __shared__ int s_expanded_tiles[kBuildTileListsThreadsPerGroup];
    __shared__ int s_expanded_tile_count;
    __shared__ int s_total_tiles_count;
    __shared__ int s_cumulated_expanded_tile_count;
    __shared__ bool s_has_pending_tiles;
    __shared__ int64_t s_tile_keys[kBuildTileListsThreadsPerGroup];
    __shared__ int32_t s_tile_values[kBuildTileListsThreadsPerGroup];
    __shared__ int32_t s_tile_index;
    __shared__ int32_t s_write_tile_start_index;
    __shared__ int32_t s_write_tile_end_index;

    // 初始化共享计数器
    if (threadIdx.x == 0) {
        s_tile_index = 0;
        s_has_pending_tiles = false;
    }

    __syncthreads();

    // 持久化循环：持续处理高斯直到处理完所有
    for (;;) {
        // 第一个warp从全局获取一批高斯（32个）
        if (threadIdx.x < kWarpSize) {
            if (threadIdx.x == 0) {
                auto splat_counter = atomicAdd(&g_splat_counter, kWarpSize);
                s_splat_start_index = glm::min(splat_counter, g_global_args.splat_count);
                auto splat_end_index = glm::min(splat_counter + kWarpSize, g_global_args.splat_count);
                s_splat_count = splat_end_index - s_splat_start_index;
                s_expanded_tile_count = 0;
                s_cumulated_expanded_tile_count = 0;
            }
        }

        __syncthreads();

        // 没有更多高斯要处理，退出
        if (s_splat_count == 0) {
            return;
        }

        auto tile_count = 0;

        // 第一个warp加载高斯数据，计算每个高斯覆盖哪些tiles
        if (threadIdx.x < kWarpSize) {
            if (threadIdx.x < s_splat_count) {
                auto src_index = s_splat_start_index + threadIdx.x;
                auto pos_xy_data = load_ro(&g_global_args.position_xy_clip[src_index]);
                auto ellipse_data = load_ro(&g_global_args.screen_ellipses[src_index]);

                Ellipse ellipse;
                ellipse.center = float2_to_glm(pos_xy_data);
                ellipse.cos_sin = glm::vec2(ellipse_data.x, ellipse_data.y);
                ellipse.extent = glm::vec2(ellipse_data.z, ellipse_data.w);

                Rect aabb = get_aabb_rect(ellipse);

                // 转换到tile坐标：clip空间 [-1,1] -> tile坐标 [0, kTilesPerScreen]
                auto tiles_rect_float = (glm::vec4(aabb.min, aabb.max) + 1.0f) * 0.5f * (float)kTilesPerScreen;

                // 裁剪到屏幕范围内
                auto tiles_rect = glm::ivec4(
                    glm::clamp((int32_t)glm::floor(tiles_rect_float.x), 0, kTilesPerScreen),
                    glm::clamp((int32_t)glm::floor(tiles_rect_float.y), 0, kTilesPerScreen),
                    glm::clamp((int32_t)glm::ceil(tiles_rect_float.z), 0, kTilesPerScreen),
                    glm::clamp((int32_t)glm::ceil(tiles_rect_float.w), 0, kTilesPerScreen));

                tiles_rect.z -= tiles_rect.x;
                tiles_rect.w -= tiles_rect.y;
                tile_count = glm::max(0, tiles_rect.z * tiles_rect.w);

                // 保存到共享内存
                s_depths[threadIdx.x] = load_ro(&g_global_args.position_z_clip[src_index]);
                s_tiles_rects[threadIdx.x] = int4{tiles_rect.x, tiles_rect.y, tiles_rect.z, tiles_rect.w};
                s_screen_ellipse[threadIdx.x] = ellipse_data;
                s_position_xy_clip[threadIdx.x] = pos_xy_data;
            }

            // 扫描计算每个线程需要写入的起始位置
            auto has_tiles = __ballot_sync(kWarpMask, tile_count != 0) != 0;
            if (has_tiles) {
                auto inclusive_scan = tile_count;
#pragma unroll
                for (auto delta = 1u; delta <= kWarpHalfSize; delta <<= 1u) {
                    auto n = __shfl_up_sync(kWarpMask, inclusive_scan, delta);
                    if (threadIdx.x >= delta) {
                        inclusive_scan += n;
                    }
                }
                s_tile_exclusive_scan[threadIdx.x] = inclusive_scan - tile_count;
            }
        }

        __syncthreads();

        if (s_expanded_tile_count == 0) {
            continue;
        }

        // 每个线程处理一个tile，将tile添加到共享内存
        for (;;) {
            if (threadIdx.x < s_expanded_tile_count) {
                auto splat_idx = s_expanded_tiles[threadIdx.x];
                auto tiles_rect = int4_to_glm(s_tiles_rects[splat_idx]);
                auto local_tile_idx = 
                    threadIdx.x - s_tile_exclusive_scan[splat_idx] + s_cumulated_expanded_tile_count - s_expanded_tile_count;
                auto local_coord = glm::ivec2(local_tile_idx % tiles_rect.z, local_tile_idx / tiles_rect.w);
                auto global_coord = glm::ivec2(tiles_rect.x, tiles_rect.y) + local_coord;
                constexpr float tile_size_normalized = (float)kTileSize / (float)kScreenSize;
                constexpr float tile_clip_size = tile_size_normalized * 2.0f;
                auto tile_clip = (glm::vec2)(global_coord) * tile_clip_size - 1.0f;

                Rect tile_rect;
                tile_rect.min = tile_clip;
                tile_rect.max = tile_clip + glm::vec2(tile_clip_size, tile_clip_size);

                Ellipse ellipse;
                ellipse.center = float2_to_glm(s_position_xy_clip[splat_idx]);
                ellipse.cos_sin = glm::vec2(s_screen_ellipse[splat_idx].x, s_screen_ellipse[splat_idx].y);
                ellipse.extent = glm::vec2(s_screen_ellipse[splat_idx].z, s_screen_ellipse[splat_idx].w);

                if (ellipse_rect_overlap(ellipse, tile_rect)) {
                    auto local_insert_idx = atomicAdd(&s_tile_index, 1);
                    auto global_tile_idx = global_coord.y * kTilesPerScreen + global_coord.x;
                    s_tile_keys[local_insert_idx] = get_sort_key(global_tile_idx, s_depths[splat_idx]);
                    s_tile_values[local_insert_idx] = s_splat_start_index + splat_idx;
                }
            }

            __syncthreads();

            if (s_tile_index > 0) {
                if (threadIdx.x == 0) {
                    auto tile_counter = atomicAdd(&g_tile_counter, s_tile_index);
                    s_write_tile_start_index = glm::min(tile_counter, g_tile_list_args.capacity);
                    s_write_tile_end_index = glm::min(tile_counter + s_tile_index, g_tile_list_args.capacity);
                    s_tile_index = 0;
                }
                __syncthreads();

                if (s_write_tile_end_index == g_tile_list_args.capacity) {
                    return;
                }

                if (threadIdx.x < s_write_tile_end_index - s_write_tile_start_index) {
                    auto dst_idx = s_write_tile_start_index + threadIdx.x;
                    g_tile_list_args.keys[dst_idx] = s_tile_keys[threadIdx.x];
                    g_tile_list_args.values[dst_idx] = s_tile_values[threadIdx.x];
                }
            }
            __syncthreads();

            if (!s_has_pending_tiles) {
                break;
            }
        }
    }
}

// 设置全局tile列表参数
void SetTileListArgs(TileListArgs* tile_list_args) {
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(g_tile_list_args, tile_list_args, sizeof(TileListArgs)));
}

// 构建tile列表（入口函数）
int BuildTileList(CudaTimer& timer, int num_blocks, int tile_list_capacity) {
    // 重置全局计数器
    int init_counter = 0;
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(g_splat_counter, &init_counter, sizeof(int32_t), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(g_tile_counter, &init_counter, sizeof(int32_t), 0, cudaMemcpyHostToDevice));

    const auto block_dim = dim3(kBuildTileListsThreadsPerGroup);
    const auto grid_dim = dim3(num_blocks);

    timer.Start();
    build_tile_list_kernel<<<grid_dim, block_dim>>>();
    timer.Stop();

    CHECK_CUDA_ERROR(cudaGetLastError());
    if (cudaGetLastError() != cudaSuccess) {
        printf("kernel error build_tile_list\n");
    }

    CHECK_CUDA_ERROR(cudaMemcpyFromSymbol(&init_counter, g_tile_counter, sizeof(int32_t), 0, cudaMemcpyDeviceToHost));
    return glm::min(init_counter, tile_list_capacity);
}

// 对tile列表按深度排序（使用cub库）
void SortTileList(CudaTimer& timer,
                  int tile_list_size,
                  void*& device_temp_storage,
                  size_t& temp_storage_size,
                  DoubleBuffer<uint64_t>& keys,
                  DoubleBuffer<int32_t>& values) {
    timer.Start();

    // 只需要对低32位（深度）排序，高32位是tile索引已经有序
    constexpr int begin_bit = 0;
    constexpr int end_bit = 32 + 12;

    cub::DoubleBuffer<uint64_t> cub_keys(keys.current(), keys.alternate());
    cub::DoubleBuffer<int32_t> cub_values(values.current(), values.alternate());

    // 第一次调用计算需要的临时存储空间大小
    size_t required_temp_size;
    cub::DeviceRadixSort::SortPairs(
        (void*) nullptr, required_temp_size, cub_keys, cub_values, tile_list_size, begin_bit, end_bit);

    if (required_temp_size > temp_storage_size) {
        if (device_temp_storage != nullptr) {
            CHECK_CUDA_ERROR(cudaFree(device_temp_storage));
        }
        CHECK_CUDA_ERROR(cudaMalloc((void**)&device_temp_storage, required_temp_size));
        temp_storage_size = required_temp_size;
    }

    // 实际排序
    cub::DeviceRadixSort::SortPairs(
        device_temp_storage, temp_storage_size, cub_keys, cub_values, tile_list_size, begin_bit, end_bit);

    // 更新双缓冲选择器
    keys = DoubleBuffer<uint64_t>(cub_keys.Current(), cub_keys.Alternate());
    values = DoubleBuffer<int32_t>(cub_values.Current(), cub_values.Alternate());

    timer.Stop();

    if (cudaGetLastError() != cudaSuccess) {
        printf("error sort_tile_list\n");
    }
}

// 计算每个tile的高斯范围：记录每个tile在排序数组中的起始和结束位置
__global__ void evaluate_tile_ranges_kernel() {
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index == 0) {
        auto first_key = load_ro(&g_tile_list_args.keys[0]);
        auto first_tile_index = first_key >> 32;
        g_global_args.tile_range[first_tile_index * 2].x = 0;

        auto last_key = load_ro(&g_tile_list_args.keys[g_tile_list_args.size - 1]);
        auto last_tile_index = last_key >> 32;
        g_global_args.tile_range[last_tile_index * 2 + 1].x = g_tile_list_args.size;
    } else if (index < g_tile_list_args.size) {
        // 和前一个元素比较，如果tile索引变化，记录范围
        auto prev_key = load_ro(&g_tile_list_args.keys[index - 1]);
        auto prev_tile = prev_key >> 32;
        auto key = load_ro(&g_tile_list_args.keys[index]);
        auto tile = key >> 32;

        if (tile != prev_tile) {
            g_global_args.tile_range[prev_tile * 2 + 1].x = index;
            g_global_args.tile_range[tile * 2].x = index;
        }
    }
}

// 计算tile范围（入口函数）
void EvaluateTileRange(CudaTimer& timer, int tile_list_size) {
    constexpr int thread_per_block = 256;
    const int num_blocks = (tile_list_size + thread_per_block - 1) / thread_per_block;
    const dim3 block_dim = dim3(thread_per_block);
    const dim3 grid_dim = dim3(num_blocks);

    timer.Start();
    evaluate_tile_ranges_kernel<<<grid_dim, block_dim>>>();
    timer.Stop();

    if (cudaGetLastError() != cudaSuccess) {
        printf("kernel error evaluate_tile_range\n");
    }
}

// 光栅化tile内核：每个tile一个block，每个线程处理tile内一个像素
__global__ void rasterize_tiles_kernel() {
    __shared__ int32_t s_first_splat_index;
    __shared__ int32_t s_splats_count;
    __shared__ float4 s_colors[kWarpSize];
    __shared__ float4 s_conics[kWarpSize];
    __shared__ float2 s_centers[kWarpSize];

    // 第一个线程读取当前tile的高斯范围
    if (threadIdx.x == 0) {
        auto tile_range_start = load_ro(&g_global_args.tile_range[blockIdx.x * 2]);
        auto tile_range_end = load_ro(&g_global_args.tile_range[blockIdx.x * 2 + 1]);
        s_first_splat_index = tile_range_start.x;
        auto end_splat_index = tile_range_end.x;
        s_splats_count = end_splat_index - s_first_splat_index;
    }

    __syncthreads();

    // 如果tile没有高斯，直接退出
    if (s_splats_count == 0) {
        return;
    }

    // 计算当前线程处理哪个像素
    const auto pixel_coord_tile = glm::ivec2(threadIdx.x % kTileSize, threadIdx.x / kTileSize);
    const auto pixel_coord_global =
        glm::ivec2(blockIdx.x % kTilesPerScreen, blockIdx.x / kTilesPerScreen) * kTileSize + pixel_coord_tile;

    // 转换到clip空间 [-1, 1]
    const auto clip_coords = (glm::vec2)(pixel_coord_global) * (2.0f / (float)kScreenSize) - 1.0f;

    // 初始化颜色和透明度
    auto color = glm::vec3(0.0f);
    auto transmittance = 1.0f;

    // 分块处理：每次加载32个高斯到共享内存，处理完再加载下一批
    while (s_splats_count > 0) {
        auto batch_count = glm::min(s_splats_count, kWarpSize);

        __syncthreads();

        // 加载当前批次的高斯数据到共享内存
        if (threadIdx.x < batch_count) {
            auto src_idx = g_tile_list_args.values[s_first_splat_index + threadIdx.x];
            s_centers[threadIdx.x] = load_ro(&g_global_args.position_xy_clip[src_idx]);
            auto conic_data = load_ro(&g_global_args.conics[src_idx]);
            auto color_data = load_ro(&g_global_args.colors[src_idx]);
            s_colors[threadIdx.x] = float4{color_data.x, color_data.y, color_data.z, color_data.w};
            s_conics[threadIdx.x] = float4{conic_data.x, conic_data.y, conic_data.z, 0.0f};
        }

        __syncthreads();

        // 光栅化当前批次的每个高斯
        for (int i = 0; i != batch_count; ++i) {
            auto d = clip_coords - float2_to_glm(s_centers[i]);
            float4 splat_color = s_colors[i];
            float3 conic = make_float3(s_conics[i].x, s_conics[i].y, s_conics[i].z);

            // 计算指数：exp(-0.5 * d^T * conic * d)
            float dx = conic.x * d.x * d.x + conic.z * d.y * d.y + 2.0f * conic.y * d.x * d.y;
            float density = __expf(-0.5f * dx);
            float alpha = splat_color.w * __saturatef(density);

            // 体积渲染：增量混合
            color += glm::vec3(splat_color.x, splat_color.y, splat_color.z) * transmittance * alpha;
            transmittance *= (1.0f - alpha);
        }

        // 第一个线程更新计数器，准备下一批次
        if (threadIdx.x == 0) {
            s_first_splat_index += batch_count;
            s_splats_count -= batch_count;
        }

        // 如果所有像素的透明度都小于阈值，提前停止渲染
        unsigned int ballot = __ballot_sync(0xffffffff, transmittance > 0.02f);
        int pop_count = __popc(ballot);
        if (pop_count == 0) {
            break;
        }

        __syncthreads();
    }

    // 输出颜色：量化到uchar4
    uchar4 quantized_color;
    quantized_color.x = (unsigned char)(glm::clamp(color.x, 0.0f, 1.0f) * 255.0f);
    quantized_color.y = (unsigned char)(glm::clamp(color.y, 0.0f, 1.0f) * 255.0f);
    quantized_color.z = (unsigned char)(glm::clamp(color.z, 0.0f, 1.0f) * 255.0f);
    quantized_color.w = 255;

    // 写入输出图像
    auto write_idx = pixel_coord_global.y * kScreenSize + pixel_coord_global.x;
    g_global_args.back_buffer[write_idx] = quantized_color;
}

// 光栅化所有tile（入口函数）
void RasterizeTiles(CudaTimer& timer) {
    // 一个block处理一个tile，每个线程处理一个像素
    constexpr int thread_per_block = kTileSize * kTileSize;
    const auto block_dim = dim3(thread_per_block);
    const auto grid_dim = dim3(kTotalTiles);

    timer.Start();
    rasterize_tiles_kernel<<<grid_dim, block_dim>>>();
    timer.Stop();

    if (cudaGetLastError() != cudaSuccess) {
        printf("kernel error rasterize_tiles\n");
    }
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

} // namespace gauss_render
