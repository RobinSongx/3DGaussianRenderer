#pragma once
#define GLM_FORCE_CUDA
#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <cuda_runtime_api.h>
#include <cstdint>
#include "consts.h"
#include "device_buffer.cu"
#include "utilities.h"
#include <assert.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>

struct CameraData
{
    glm::mat4 viewProjection;
    glm::mat4 projection;
    glm::mat4 view;
    glm::vec2 fovCotangent;
    glm::vec2 depthScaleBias;
    glm::vec3 position;
    float aspect;
};

struct Rect
{
    __host__ __device__ Rect() : min(0.0f), max(0.0f) {}
    glm::vec2 min;
    glm::vec2 max;

    __device__ __host__ glm::vec2 getCenter() const
    {
        return min + (max - min) * 0.5f;
    }
};

struct Ellipse
{
    __host__ __device__ Ellipse() : center(0.0f), extent(0.0f), cosSin(0.0f) {}
    glm::vec2 center;
    glm::vec2 extent;
    glm::vec2 cosSin;

    __device__ __host__ glm::vec2 getPrincipalAxis() const
    {
        return cosSin * extent.x;
    }
    __device__ __host__ glm::vec2 getMinorAxis() const
    {
        return glm::vec2(cosSin.y, -cosSin.x) * extent.y;
    }
};

struct GlobalArgs
{
    CameraData cameraData;
    float4* position;
    float4* scaleAndRotation;
    float4* color;
    int sphericalHarmonicsDegree;
    int sphericalHarmonicsCount;
    float* sphericalHarmonics;
    float2* positionClipSpaceXY;
    float* positionClipSpaceZ;
    float4* conic;
    float4* screenEllipse;
    int32_t splatCount;
    uchar4* backBuffer;
    int32_t* tileRange;
};

struct TileListArgs
{
    uint64_t* keys;
    int32_t* values;
    int32_t size;
    int32_t capacity;
};

template <typename T>
struct DoubleBuffer
{
  private:
    T* m_Buffers[2];
    int m_Selector;

  public:
    DoubleBuffer(T* current, T* alternate)
    {
        m_Selector = 0;
        m_Buffers[0] = current;
        m_Buffers[1] = alternate;
    }

    T* current() const
    {
        return m_Buffers[m_Selector];
    }

    T* alternate() const
    {
        return m_Buffers[m_Selector ^ 1];
    }

    int selector() const
    {
        return m_Selector;
    }
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
