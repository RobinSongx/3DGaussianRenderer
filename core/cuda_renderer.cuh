#ifndef GAUSS_RENDER_CUDA_RENDERER_CUH_
#define GAUSS_RENDER_CUDA_RENDERER_CUH_

#include "types.h"
#include "camera.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>

namespace gauss_render {

struct CameraData {
    glm::vec3 position;
    float aspect;
    glm::mat4 projection;
    glm::mat4 view;
    glm::mat4 viewProjection;
    glm::vec2 fovCotangent;
    glm::vec2 depthScaleBias;
};

template <typename T>
struct DoubleBuffer {
  private:
    T* m_Buffers[2];
    int m_Selector;

  public:
    DoubleBuffer(T* current, T* alternate) {
        m_Selector = 0;
        m_Buffers[0] = current;
        m_Buffers[1] = alternate;
    }

    T* current() const {
        return m_Buffers[m_Selector];
    }

    T* alternate() const {
        return m_Buffers[m_Selector ^ 1];
    }

    int selector() const {
        return m_Selector;
    }
};

struct GlobalArgs {
    int32_t splatCount;
    int sphericalHarmonicsDegree;
    int sphericalHarmonicsCount;
    float4* position;
    float4* scaleAndRotation;
    float4* color;
    float4* conic;
    float4* screenEllipse;
    float2* positionClipSpaceXY;
    float* positionClipSpaceZ;
    float* sphericalHarmonics;
    int32_t* tileRange;
    uchar4* backBuffer;
    CameraData cameraData;
};

struct TileListArgs {
    uint64_t* keys;
    int32_t* values;
    int32_t size;
    int32_t capacity;
};

void setGlobalArgs(GlobalArgs* globalArgs);
void setTileListArgs(TileListArgs* tileListArgs);

void evaluateSphericalHarmonics(CudaTimer& timer, int32_t count);
void evaluateSplatClipData(CudaTimer& timer, int32_t count);
int32_t buildTileList(CudaTimer& timer, int32_t numBlocks, int32_t tileListCapacity);
void sortTileList(CudaTimer& timer,
                  int32_t tileListSize,
                  void*& deviceTempStorage,
                  size_t& tempStorageSizeInBytes,
                  DoubleBuffer<uint64_t>& keys,
                  DoubleBuffer<int32_t>& values);
void evaluateTileRange(CudaTimer& timer, int32_t tileListSize);
void rasterizeTile(CudaTimer& timer);

void ClearScreen(uchar4* buffer, int width, int height);

} // namespace gauss_render

#endif // GAUSS_RENDER_CUDA_RENDERER_CUH_
