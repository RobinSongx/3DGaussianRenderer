#ifndef GAUSS_RENDER_TYPES_H_
#define GAUSS_RENDER_TYPES_H_

#include <glm/glm.hpp>
#include <cstdint>

namespace gauss_render {

// 每个tile的大小（像素）
constexpr int kTileSize = 16;
// 默认屏幕大小
constexpr int kScreenSize = 1024;
// 默认屏幕宽度
constexpr int kDefaultScreenWidth = 1024;
// 默认屏幕高度
constexpr int kDefaultScreenHeight = 1024;
// 球谐计算每个block处理的高斯数量（用于内存对齐）
constexpr int kDefaultShBlockSize = 256;
// 每一行/列有多少个tile
constexpr int kTilesPerScreen = kScreenSize / kTileSize;
// 总的tile数量
constexpr int kTotalTiles = kTilesPerScreen * kTilesPerScreen;

// 渲染性能统计
// 记录每个渲染阶段的耗时（毫秒）
struct RenderStats {
    double sh_eval_ms = 0.0;       // 球谐评估阶段耗时
    double project_ms = 0.0;       // 投影阶段耗时
    double build_tile_ms = 0.0;    // 构建tile列表阶段耗时
    double sort_ms = 0.0;          // 排序阶段耗时
    double raster_ms = 0.0;        // 光栅化阶段耗时
    double total_ms = 0.0;         // 总耗时
};

// 四元数量化：将4个float压缩到一个uint32中
// 每个分量8位，因为四元数分量在[-1, 1]范围内，可以映射到[0, 255]
// 节省50%内存
inline uint32_t QuantizeQuaternion(float q[4]) {
    uint32_t result = 0;
    for (int i = 0; i < 4; ++i) {
        uint32_t v = static_cast<uint32_t>((q[i] + 1.0f) * 0.5f * 255.0f);
        result |= (v & 0xFFU) << (i * 8);
    }
    return result;
}

// 四元数量化解码：从uint32还原出4个float分量
inline void DequantizeQuaternion(uint32_t packed, float q[4]) {
    for (int i = 0; i < 4; ++i) {
        uint32_t v = (packed >> (i * 8)) & 0xFFU;
        q[i] = static_cast<float>(v) / 255.0f * 2.0f - 1.0f;
    }
}

} // namespace gauss_render

#endif // GAUSS_RENDER_TYPES_H_
