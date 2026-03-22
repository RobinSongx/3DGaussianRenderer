#ifndef GAUSS_RENDER_TYPES_H_
#define GAUSS_RENDER_TYPES_H_

#include <glm/glm.hpp>
#include <cstdint>

namespace gauss_render {

constexpr int kTileSize = 16;
constexpr int kScreenSize = 1024;
constexpr int kDefaultScreenWidth = 1024;
constexpr int kDefaultScreenHeight = 1024;
constexpr int kDefaultShBlockSize = 256;
constexpr int kTilesPerScreen = kScreenSize / kTileSize;
constexpr int kTotalTiles = kTilesPerScreen * kTilesPerScreen;

struct RenderStats {
    double sh_eval_ms = 0.0;
    double project_ms = 0.0;
    double build_tile_ms = 0.0;
    double sort_ms = 0.0;
    double raster_ms = 0.0;
    double total_ms = 0.0;
};

inline uint32_t QuantizeQuaternion(float q[4]) {
    uint32_t result = 0;
    for (int i = 0; i < 4; ++i) {
        uint32_t v = static_cast<uint32_t>((q[i] + 1.0f) * 0.5f * 255.0f);
        result |= (v & 0xFFU) << (i * 8);
    }
    return result;
}

inline void DequantizeQuaternion(uint32_t packed, float q[4]) {
    for (int i = 0; i < 4; ++i) {
        uint32_t v = (packed >> (i * 8)) & 0xFFU;
        q[i] = static_cast<float>(v) / 255.0f * 2.0f - 1.0f;
    }
}

} // namespace gauss_render

#endif // GAUSS_RENDER_TYPES_H_
