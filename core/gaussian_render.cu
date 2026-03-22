#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include "cuda_error_check.cpp"
#include "gaussian_render.cuh"

template <typename T>
__device__ T loadReadOnly(T* ptr)
{
    return __ldg(reinterpret_cast<const T*>(ptr));
}

__device__ __host__ __forceinline__ glm::vec2 builtinToGlmVec2(const float2 v)
{
    return glm::vec2(v.x, v.y);
}

__device__ __host__ __forceinline__ glm::vec4 builtinToGlmVec4(const float4 v)
{
    return glm::vec4(v.x, v.y, v.z, v.w);
}

__device__ __host__ __forceinline__ glm::ivec4 builtinToGlmVec4i(const int4 v)
{
    return glm::ivec4(v.x, v.y, v.z, v.w);
}

__device__ __managed__ int32_t g_SplatCounter;
__device__ __managed__ int32_t g_TileCounter;

__constant__ GlobalArgs g_GlobalArgs;
__constant__ TileListArgs g_TileListArgs;

void setGlobalArgs(GlobalArgs* globalArgs)
{
    checkCudaErrors(cudaMemcpyToSymbol(g_GlobalArgs, globalArgs, sizeof(GlobalArgs), 0, cudaMemcpyHostToDevice));
}

void setTileListArgs(TileListArgs* tileListArgs)
{
    checkCudaErrors(cudaMemcpyToSymbol(g_TileListArgs, tileListArgs, sizeof(TileListArgs), 0, cudaMemcpyHostToDevice));
}

__device__ glm::vec4 decodeVec4(uint32_t v)
{
    return glm::vec4((v >> 24u) & 0xFF, (v >> 16u) & 0xFF, (v >> 8u) & 0xFF, v & 0xFF) / 255.0f;
}

__device__ glm::vec3 sphericalHarmonics(const int l, const glm::vec3& dir, const float* sh, const int stride)
{
    const auto x = dir.x;
    const auto y = dir.y;
    const auto z = dir.z;
    auto result = glm::vec3(0.0f);

    auto sh0 = glm::vec3(loadReadOnly(&sh[0 * stride]), loadReadOnly(&sh[1 * stride]), loadReadOnly(&sh[2 * stride]));

    result +=
        (0.282094792F) * sh0;

    if (l > 0)
    {
        auto sh1 = glm::vec3(loadReadOnly(&sh[3 * stride]), loadReadOnly(&sh[4 * stride]), loadReadOnly(&sh[5 * stride]));
        auto sh2 = glm::vec3(loadReadOnly(&sh[6 * stride]), loadReadOnly(&sh[7 * stride]), loadReadOnly(&sh[8 * stride]));
        auto sh3 = glm::vec3(loadReadOnly(&sh[9 * stride]), loadReadOnly(&sh[10 * stride]), loadReadOnly(&sh[11 * stride]));

        result +=
            (0.488602512F*y) * sh1 +
            (0.488602512F*z) * sh2 +
            (0.488602512F*x) * sh3;

        if (l > 1)
        {
            auto xx = x * x;
            auto yy = y * y;
            auto zz = z * z;

            auto sh4 = glm::vec3(loadReadOnly(&sh[12 * stride]), loadReadOnly(&sh[13 * stride]), loadReadOnly(&sh[14 * stride]));
            auto sh5 = glm::vec3(loadReadOnly(&sh[15 * stride]), loadReadOnly(&sh[16 * stride]), loadReadOnly(&sh[17 * stride]));
            auto sh6 = glm::vec3(loadReadOnly(&sh[18 * stride]), loadReadOnly(&sh[19 * stride]), loadReadOnly(&sh[20 * stride]));
            auto sh7 = glm::vec3(loadReadOnly(&sh[21 * stride]), loadReadOnly(&sh[22 * stride]), loadReadOnly(&sh[23 * stride]));
            auto sh8 = glm::vec3(loadReadOnly(&sh[24 * stride]), loadReadOnly(&sh[25 * stride]), loadReadOnly(&sh[26 * stride]));

            result +=
                (1.09254843F*x*y) * sh4 +
                (1.09254843F*y*z) * sh5 +
                (-0.946174696F*xx - 0.946174696F*yy + 0.630783131F) * sh6 +
                (1.09254843F*x*z) * sh7 +
                (0.546274215F*(x - y)*(x + y)) * sh8;
        }
    }
    return glm::clamp(result + 0.5f, 0.0f, 1.0f);
}

__global__ void evaluateSphericalHarmonicsKernel()
{
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < g_GlobalArgs.splatCount)
    {
        auto positionData = loadReadOnly(&g_GlobalArgs.position[index]);

        auto splatPosition = glm::vec3(positionData.x, positionData.y, positionData.z);

        auto opacity = positionData.w;

        auto rayDir = glm::normalize(g_GlobalArgs.cameraData.position - splatPosition);

        auto shIdx = blockIdx.x * blockDim.x * g_GlobalArgs.sphericalHarmonicsCount + threadIdx.x;

        auto shColor = sphericalHarmonics(
            g_GlobalArgs.sphericalHarmonicsDegree, rayDir, &g_GlobalArgs.sphericalHarmonics[shIdx], blockDim.x);

        g_GlobalArgs.color[index] = float4{shColor.x, shColor.y, shColor.z, opacity};
    }
}

void evaluateSphericalHarmonics(CudaTimer& timer, int32_t count)
{
    constexpr int32_t threadPerBlock{256};
    const int32_t numBlocks{(count + threadPerBlock - 1) / threadPerBlock};
    const auto dimBlock{dim3(threadPerBlock)};
    const auto dimGrid{dim3(numBlocks)};

    timer.start();
    evaluateSphericalHarmonicsKernel<<<dimGrid, dimBlock>>>();
    timer.stop();

    if (cudaGetLastError() != cudaSuccess)
    {
        printf("kernel error evaluateSphericalHarmonics\n");
    }
}

__global__ void evaluateSplatClipDataKernel()
{
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < g_GlobalArgs.splatCount)
    {
        auto positionData = loadReadOnly(&g_GlobalArgs.position[index]);
        auto scaleAndRotationData = loadReadOnly(&g_GlobalArgs.scaleAndRotation[index]);

        auto position = glm::vec3(positionData.x, positionData.y, positionData.z);

        auto scale3x3 = glm::mat3(0);
        scale3x3[0][0] = scaleAndRotationData.x;
        scale3x3[1][1] = scaleAndRotationData.y;
        scale3x3[2][2] = scaleAndRotationData.z;

        auto rotationValue = decodeVec4(reinterpret_cast<uint32_t&>(scaleAndRotationData.w)) * 2.0f - 1.0f;
        auto rotation = glm::quat(rotationValue.w, rotationValue.x, rotationValue.y, rotationValue.z);

        auto RS = glm::mat3_cast(rotation) * scale3x3;
        auto splatCovariance = RS * glm::transpose(RS);

        auto viewPosition = (glm::vec3)(g_GlobalArgs.cameraData.view * glm::vec4(position, 1));
        auto fovCotangent = g_GlobalArgs.cameraData.fovCotangent;
        auto depthScaleBias = g_GlobalArgs.cameraData.depthScaleBias;

        auto zRcp = 1.0f / viewPosition.z;
        auto zRcpSqr = zRcp * zRcp;
        auto scaleX = -fovCotangent.x * zRcp;
        auto scaleY = -fovCotangent.y * zRcp;
        auto fovCotangentTimesTrs = fovCotangent * glm::vec2(viewPosition.x, viewPosition.y);
        auto shearX = fovCotangentTimesTrs.x * zRcpSqr;
        auto shearY = fovCotangentTimesTrs.y * zRcpSqr;
        auto translationX = -fovCotangentTimesTrs.x * zRcp;
        auto translationY = -fovCotangentTimesTrs.y * zRcp;

        auto affineProjection = glm::mat4x4(1);
        affineProjection[0][0] = scaleX;
        affineProjection[1][1] = scaleY;
        affineProjection[2][2] = depthScaleBias.x;
        affineProjection[2][0] = shearX;
        affineProjection[2][1] = shearY;
        affineProjection[3][0] = translationX;
        affineProjection[3][1] = translationY;
        affineProjection[3][2] = depthScaleBias.y;

        auto viewProjection = (glm::mat3) affineProjection * (glm::mat3) g_GlobalArgs.cameraData.view;
        auto clipCovariance = viewProjection * glm::transpose(splatCovariance) * glm::transpose(viewProjection);
        auto clipPosition = (glm::vec3)(affineProjection * glm::vec4(viewPosition, 1));

        constexpr float k_Pi{3.14159265359};
        constexpr float k_TexelSizeClip{2.0f / (float) k_ScreenSize};
        constexpr float k_TraceBump{(1.0f / k_Pi) * k_TexelSizeClip * k_TexelSizeClip};
        clipCovariance[0][0] += k_TraceBump;
        clipCovariance[1][1] += k_TraceBump;

        auto det = clipCovariance[0][0] * clipCovariance[1][1] - clipCovariance[1][0] * clipCovariance[1][0];
        auto mid = 0.5f * (clipCovariance[0][0] + clipCovariance[1][1]);
        constexpr float epsilon = 1e-12f;
        auto radius = glm::sqrt(glm::max(epsilon, mid * mid - det));
        auto lambda0 = mid + radius;
        auto lambda1 = glm::max(0.0f, mid - radius);

        auto eigenVector0 = glm::normalize(glm::vec2(clipCovariance[1][0], lambda0 - clipCovariance[0][0]));
        auto eigenVector1 = glm::normalize(glm::vec2(eigenVector0.y / g_GlobalArgs.cameraData.aspect, -eigenVector0.x));

        auto angle = glm::atan(eigenVector0.y, eigenVector0.x);
        auto extent = glm::sqrt(glm::vec2(lambda0, lambda1)) * 3.0f;
        auto invDet = 1.0f / glm::max(epsilon, det);
        auto conic = glm::vec3(clipCovariance[1][1], -clipCovariance[1][0], clipCovariance[0][0]) * invDet;

        auto edge = glm::step(glm::vec3(-1.0f), clipPosition) * glm::step(clipPosition, glm::vec3(1.0f));
        auto isVisible = edge.x * edge.y * edge.z * glm::step(0.0f, lambda1);
        clipPosition = glm::mix(glm::vec3(-128.0f), clipPosition, isVisible);
        extent *= isVisible;

        g_GlobalArgs.positionClipSpaceXY[index] = float2{clipPosition.x, clipPosition.y};
        g_GlobalArgs.positionClipSpaceZ[index] = clipPosition.z;
        g_GlobalArgs.screenEllipse[index] = float4{glm::cos(angle), glm::sin(angle), extent.x, extent.y};
        g_GlobalArgs.conic[index] = float4{conic.x, conic.y, conic.z, 0.0f};
    }
}

void evaluateSplatClipData(CudaTimer& timer, int32_t count)
{
    constexpr int32_t threadPerBlock{256};
    const int32_t numBlocks{(count + threadPerBlock - 1) / threadPerBlock};
    const auto dimBlock{dim3(threadPerBlock)};
    const auto dimGrid{dim3(numBlocks)};

    timer.start();
    evaluateSplatClipDataKernel<<<dimGrid, dimBlock>>>();
    timer.stop();

    if (cudaGetLastError() != cudaSuccess)
    {
        printf("kernel error evaluateSplatsClipData\n");
    }
}

__device__ __forceinline__ glm::vec2 convertToEllipseCoordinates(const struct Ellipse& ellipse, const glm::vec2& point)
{
    auto delta = point - ellipse.center;
    return glm::vec2((delta.x * ellipse.cosSin.x + delta.y * ellipse.cosSin.y) / ellipse.extent.x,
                     (delta.y * ellipse.cosSin.x - delta.x * ellipse.cosSin.y) / ellipse.extent.y);
}

__device__ __forceinline__ bool intersectsUnitCircle(const glm::vec2& v0, const glm::vec2& v1)
{
    auto delta = v1 - v0;
    auto lengthSqr = glm::length2(delta);
    auto t = __saturatef(glm::dot(-v0, delta) / lengthSqr);
    auto projection = v0 + t * (v1 - v0);
    return glm::length2(projection) < 1.0f;
}

__device__ __forceinline__ bool ellipseRectOverlap(const struct Ellipse& ellipse, const struct Rect& rect)
{
    auto overlaps =
        ellipse.center.x > rect.min.x &&
        ellipse.center.x < rect.max.x &&
        ellipse.center.y > rect.min.y &&
        ellipse.center.y < rect.max.y;

    overlaps |= glm::length2(convertToEllipseCoordinates(ellipse, rect.getCenter())) < 1.0f;

    glm::vec2 points[4];
    points[0] = convertToEllipseCoordinates(ellipse, rect.min);
    points[1] = convertToEllipseCoordinates(ellipse, glm::vec2(rect.max.x, rect.min.y));
    points[2] = convertToEllipseCoordinates(ellipse, rect.max);
    points[3] = convertToEllipseCoordinates(ellipse, glm::vec2(rect.min.x, rect.max.y));

    overlaps |= intersectsUnitCircle(points[0], points[1]);
    overlaps |= intersectsUnitCircle(points[1], points[2]);
    overlaps |= intersectsUnitCircle(points[2], points[3]);
    overlaps |= intersectsUnitCircle(points[3], points[0]);

    return overlaps;
}

__device__ __forceinline__ struct Rect getAABBRect(const struct Ellipse& ellipse)
{
    auto right = ellipse.getPrincipalAxis();
    auto up = ellipse.getMinorAxis();

    Rect rect;
    rect.min = glm::vec2(1e12f, 1e12f);
    rect.max = glm::vec2(-1e12f, -1e12f);

#pragma unroll
    for (auto j = 0; j != 4; ++j)
    {
        auto bottomBit = j & 1;
        auto topBit = j >> 1;
        auto r = (bottomBit ^ topBit) * 2.0f - 1.0f;
        auto u = (topBit) * 2.0f - 1.0f;
        auto v = right * r + up * u;
        rect.min = glm::min(rect.min, v);
        rect.max = glm::max(rect.max, v);
    }

    rect.min += ellipse.center;
    rect.max += ellipse.center;
    return rect;
}

constexpr uint32_t k_WarpMask{0xffffffff};
constexpr int32_t k_WarpSize{32};
constexpr int32_t k_WarpHalfSize{k_WarpSize / 2};
constexpr int32_t k_BuildTileListsThreadsPerGroup{k_WarpSize * 8};
constexpr uint32_t k_MaxUint32{0xffffffff};

__device__ __forceinline__ uint64_t getKey(int32_t tileIndex, float clipDepth)
{
    auto quantizedDepth = (uint32_t) (glm::clamp((clipDepth + 1.0f) * 0.5f, 0.0f, 1.0f) * k_MaxUint32);
    return ((uint64_t) tileIndex << 32) | quantizedDepth;
}

__global__ void buildTileListKernel()
{
    __shared__ int32_t s_SplatStartIndex;
    __shared__ int32_t s_SplatCount;

    __shared__ int4 s_TilesRects[k_WarpSize];
    __shared__ float4 s_ScreenEllipse[k_WarpSize];
    __shared__ float2 s_PositionClipSpaceXY[k_WarpSize];
    __shared__ float s_Depths[k_WarpSize];
    __shared__ int s_TileExclusiveScan[k_WarpSize];

    __shared__ int s_ExpandedTiles[k_BuildTileListsThreadsPerGroup];
    __shared__ int s_ExpandedTileCount;
    __shared__ int s_TotalTilesCount;
    __shared__ int s_CumulatedExpandedTileCount;
    __shared__ bool s_HasPendingTiles;

    __shared__ int64_t s_TileKeys[k_BuildTileListsThreadsPerGroup];
    __shared__ int32_t s_TileValues[k_BuildTileListsThreadsPerGroup];
    __shared__ int32_t s_TileIndex;
    __shared__ int32_t s_WriteTileStartIndex;
    __shared__ int32_t s_WriteTileEndIndex;

    if (threadIdx.x == 0)
    {
        s_TileIndex = 0;
        s_HasPendingTiles = false;
    }

    __syncthreads();

    for (;;)
    {
        if (threadIdx.x == 0)
        {
            auto splatCounter = atomicAdd(&g_SplatCounter, k_WarpSize);
            s_SplatStartIndex = glm::min(splatCounter, g_GlobalArgs.splatCount);
            auto splatEndIndex = glm::min(splatCounter + k_WarpSize, g_GlobalArgs.splatCount);
            s_SplatCount = splatEndIndex - s_SplatStartIndex;
            s_ExpandedTileCount = 0;
            s_CumulatedExpandedTileCount = 0;
        }

        __syncthreads();

        if (s_SplatCount == 0)
        {
            return;
        }

        auto tileCount = 0;

        if (threadIdx.x < k_WarpSize)
        {
            if (threadIdx.x < s_SplatCount)
            {
                auto srcIndex = s_SplatStartIndex + threadIdx.x;
                auto positionClipSpaceXYData = loadReadOnly(&g_GlobalArgs.positionClipSpaceXY[srcIndex]);
                auto ellipseData = loadReadOnly(&g_GlobalArgs.screenEllipse[srcIndex]);

                auto ellipse = struct Ellipse();
                ellipse.center = glm::vec2(positionClipSpaceXYData.x, positionClipSpaceXYData.y);
                ellipse.cosSin = glm::vec2(ellipseData.x, ellipseData.y);
                ellipse.extent = glm::vec2(ellipseData.z, ellipseData.w);

                auto rect = getAABBRect(ellipse);

                auto tilesRectFloat = (glm::vec4(rect.min, rect.max) + 1.0f) * 0.5f * (float) k_TilesPerScreen;

                auto tilesRect = glm::ivec4(
                    glm::clamp((int32_t) glm::floor(tilesRectFloat.x), 0, k_TilesPerScreen),
                    glm::clamp((int32_t) glm::floor(tilesRectFloat.y), 0, k_TilesPerScreen),
                    glm::clamp((int32_t) glm::ceil(tilesRectFloat.z), 0, k_TilesPerScreen),
                    glm::clamp((int32_t) glm::ceil(tilesRectFloat.w), 0, k_TilesPerScreen));

                auto tilesRectZW = glm::ivec2(tilesRect.z - tilesRect.x, tilesRect.w - tilesRect.y);

                tileCount = glm::max(0, tilesRectZW.x * tilesRectZW.y);

                s_Depths[threadIdx.x] = loadReadOnly(&g_GlobalArgs.positionClipSpaceZ[srcIndex]);
                s_TilesRects[threadIdx.x] = int4{tilesRect.x, tilesRect.y, tilesRectZW.x, tilesRectZW.y};
                s_ScreenEllipse[threadIdx.x] = ellipseData;
                s_PositionClipSpaceXY[threadIdx.x] = positionClipSpaceXYData;
            }

            auto hasTiles = __ballot_sync(k_WarpMask, tileCount != 0) != 0;

            if (hasTiles)
            {
                auto inclusiveScan = tileCount;
#pragma unroll
                for (auto delta = 1u; delta <= k_WarpHalfSize; delta <<= 1u)
                {
                    auto n = __shfl_up_sync(k_WarpMask, inclusiveScan, delta);

                    if (threadIdx.x >= delta)
                    {
                        inclusiveScan += n;
                    }
                }

                s_TileExclusiveScan[threadIdx.x] = inclusiveScan - tileCount;

                if (threadIdx.x == k_WarpSize - 1)
                {
                    s_TotalTilesCount = inclusiveScan;
                }

                auto startWriteIndex = glm::min(inclusiveScan - tileCount, k_BuildTileListsThreadsPerGroup);
                auto endWriteIndex = glm::min(inclusiveScan, k_BuildTileListsThreadsPerGroup);
                auto writeIndex = startWriteIndex;

                while (__ballot_sync(k_WarpMask, writeIndex < endWriteIndex) != 0)
                {
                    if (writeIndex < endWriteIndex)
                    {
                        s_ExpandedTiles[writeIndex] = threadIdx.x;
                        ++writeIndex;
                    }
                }

                __syncwarp();

                if (endWriteIndex - startWriteIndex > 0)
                {
                    if (endWriteIndex == s_TotalTilesCount)
                    {
                        s_HasPendingTiles = false;
                        s_ExpandedTileCount = endWriteIndex;
                        s_CumulatedExpandedTileCount = s_ExpandedTileCount;
                    }
                    else if (endWriteIndex == k_BuildTileListsThreadsPerGroup)
                    {
                        s_HasPendingTiles = true;
                        s_ExpandedTileCount = endWriteIndex;
                        s_CumulatedExpandedTileCount = s_ExpandedTileCount;
                    }
                }
            }
        }

        __syncthreads();

        if (s_ExpandedTileCount == 0)
        {
            continue;
        }

        for (;;)
        {
            if (threadIdx.x < s_ExpandedTileCount)
            {
                auto splatIndex = s_ExpandedTiles[threadIdx.x];
                auto tilesRect = builtinToGlmVec4i(s_TilesRects[splatIndex]);
                auto localTileIndex =
                    threadIdx.x - s_TileExclusiveScan[splatIndex] + s_CumulatedExpandedTileCount - s_ExpandedTileCount;

                auto localTileCoords = glm::ivec2(localTileIndex % tilesRect.z, localTileIndex / tilesRect.z);
                auto globalTileCoords = tilesRect.xy + localTileCoords;
                constexpr float tileNormalizedSize = k_TileSize / (float) k_ScreenSize;
                constexpr float tileClipSize = tileNormalizedSize * 2.0f;
                auto tileClip = (glm::vec2)(globalTileCoords) * tileClipSize - 1.0f;
                Rect tileRectClipSpace;
                tileRectClipSpace.min = tileClip;
                tileRectClipSpace.max = tileClip + glm::vec2(tileClipSize, tileClipSize);

                auto ellipse = struct Ellipse();
                ellipse.center = glm::vec2(s_PositionClipSpaceXY[splatIndex].x, s_PositionClipSpaceXY[splatIndex].y);
                ellipse.cosSin = glm::vec2(s_ScreenEllipse[splatIndex].x, s_ScreenEllipse[splatIndex].y);
                ellipse.extent = glm::vec2(s_ScreenEllipse[splatIndex].z, s_ScreenEllipse[splatIndex].w);

                if (ellipseRectOverlap(ellipse, tileRectClipSpace))
                {
                    auto localInsertionIndex = atomicAdd(&s_TileIndex, 1);
                    auto globalTileIndex = globalTileCoords.y * k_TilesPerScreen + globalTileCoords.x;
                    s_TileKeys[localInsertionIndex] = getKey(globalTileIndex, s_Depths[splatIndex]);
                    s_TileValues[localInsertionIndex] = s_SplatStartIndex + splatIndex;
                }
            }

            __syncthreads();

            if (s_TileIndex > 0)
            {
                if (threadIdx.x == 0)
                {
                    auto tileTileListCounter = atomicAdd(&g_TileCounter, s_TileIndex);
                    s_WriteTileStartIndex = glm::min(tileTileListCounter, g_TileListArgs.capacity);
                    s_WriteTileEndIndex = glm::min(tileTileListCounter + s_TileIndex, g_TileListArgs.capacity);
                    s_TileIndex = 0;
                }

                __syncthreads();

                if (s_WriteTileEndIndex == g_TileListArgs.capacity)
                {
                    return;
                }

                if (threadIdx.x < s_WriteTileEndIndex - s_WriteTileStartIndex)
                {
                    auto dstIndex = s_WriteTileStartIndex + threadIdx.x;
                    g_TileListArgs.keys[dstIndex] = s_TileKeys[threadIdx.x];
                    g_TileListArgs.values[dstIndex] = s_TileValues[threadIdx.x];
                }
            }

            __syncthreads();

            if (!s_HasPendingTiles)
            {
                break;
            }

            if (threadIdx.x < k_WarpSize)
            {
                auto exclusiveScan = s_TileExclusiveScan[threadIdx.x] - s_CumulatedExpandedTileCount;
                auto startWriteIndex = glm::clamp(exclusiveScan, 0, k_BuildTileListsThreadsPerGroup);
                auto endWriteIndex = glm::clamp(exclusiveScan + tileCount, 0, k_BuildTileListsThreadsPerGroup);
                auto writeIndex = startWriteIndex;

                while (__ballot_sync(k_WarpMask, writeIndex < endWriteIndex) != 0)
                {
                    if (writeIndex < endWriteIndex)
                    {
                        s_ExpandedTiles[writeIndex] = threadIdx.x;
                        ++writeIndex;
                    }
                }

                __syncwarp();

                if (endWriteIndex - startWriteIndex > 0)
                {
                    if (endWriteIndex == s_TotalTilesCount - s_CumulatedExpandedTileCount)
                    {
                        s_HasPendingTiles = false;
                        s_ExpandedTileCount = endWriteIndex;
                        s_CumulatedExpandedTileCount += s_ExpandedTileCount;
                    }
                    else if (endWriteIndex == k_BuildTileListsThreadsPerGroup)
                    {
                        s_HasPendingTiles = true;
                        s_ExpandedTileCount = endWriteIndex;
                        s_CumulatedExpandedTileCount += s_ExpandedTileCount;
                    }
                }
            }

            __syncthreads();
        }
    }
}

static int32_t initCounter{0};

int32_t buildTileList(CudaTimer& timer, int32_t numBlocks, int32_t tileListCapacity)
{
    initCounter = 0;
    checkCudaErrors(cudaMemcpyToSymbol(g_SplatCounter, &initCounter, sizeof(int32_t), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(g_TileCounter, &initCounter, sizeof(int32_t), 0, cudaMemcpyHostToDevice));

    const auto dimBlock = dim3(k_BuildTileListsThreadsPerGroup);
    const auto dimGrid = dim3(numBlocks);

    timer.start();
    buildTileListKernel<<<dimGrid, dimBlock>>>();
    timer.stop();

    checkCudaErrors(cudaMemcpyFromSymbol(&initCounter, g_TileCounter, sizeof(int32_t), 0, cudaMemcpyDeviceToHost));

    if (cudaGetLastError() != cudaSuccess)
    {
        printf("kernel error buildTileList\n");
    }

    return glm::min(initCounter, tileListCapacity);
}

void sortTileList(CudaTimer& timer,
                  int32_t tileListSize,
                  void*& deviceTempStorage,
                  size_t& tempStorageSizeInBytes,
                  DoubleBuffer<uint64_t>& keys,
                  DoubleBuffer<int32_t>& values)
{
    timer.start();

    constexpr int32_t beginBit{0};
    constexpr int32_t endBit{32 + 12};

    cub::DoubleBuffer<uint64_t> cubKeys(keys.current(), keys.alternate());
    cub::DoubleBuffer<int32_t> cubValues(values.current(), values.alternate());

    size_t requiredTempStorageSizeInBytes;
    cub::DeviceRadixSort::SortPairs(
        (void*) nullptr, requiredTempStorageSizeInBytes, cubKeys, cubValues, tileListSize, beginBit, endBit);

    if (requiredTempStorageSizeInBytes > tempStorageSizeInBytes)
    {
        if (deviceTempStorage != nullptr)
        {
            checkCudaErrors(cudaFree(deviceTempStorage));
        }
        checkCudaErrors(cudaMalloc((void**) &deviceTempStorage, requiredTempStorageSizeInBytes));
        tempStorageSizeInBytes = requiredTempStorageSizeInBytes;
    }

    cub::DeviceRadixSort::SortPairs(
        deviceTempStorage, tempStorageSizeInBytes, cubKeys, cubValues, tileListSize, beginBit, endBit);

    keys = DoubleBuffer<uint64_t>(cubKeys.Current(), cubKeys.Alternate());
    values = DoubleBuffer<int32_t>(cubValues.Current(), cubValues.Alternate());

    timer.stop();

    if (cudaGetLastError() != cudaSuccess)
    {
        printf("error sortTileList\n");
    }
}

__global__ void evaluateTileRangesKernel()
{
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index == 0)
    {
        auto firstKey = loadReadOnly(&g_TileListArgs.keys[0]);
        auto firstTileIndex = firstKey >> 32;
        g_GlobalArgs.tileRange[firstTileIndex * 2] = 0;

        auto lastKey = loadReadOnly(&g_TileListArgs.keys[g_TileListArgs.size - 1]);
        auto lastTileIndex = lastKey >> 32;
        g_GlobalArgs.tileRange[lastTileIndex * 2 + 1] = g_TileListArgs.size;
    }
    else if (index < g_TileListArgs.size)
    {
        auto prevKey = loadReadOnly(&g_TileListArgs.keys[index - 1]);
        auto prevTileIndex = prevKey >> 32;
        auto key = loadReadOnly(&g_TileListArgs.keys[index]);
        auto tileIndex = key >> 32;

        if (tileIndex != prevTileIndex)
        {
            g_GlobalArgs.tileRange[prevTileIndex * 2 + 1] = index;
            g_GlobalArgs.tileRange[tileIndex * 2] = index;
        }
    }
}

void evaluateTileRange(CudaTimer& timer, int32_t tileListSize)
{
    constexpr int32_t threadPerBlock{256};
    const int32_t numBlocks{(tileListSize + threadPerBlock - 1) / threadPerBlock};
    const auto dimBlock{dim3(threadPerBlock)};
    const auto dimGrid{dim3(numBlocks)};

    timer.start();
    evaluateTileRangesKernel<<<dimGrid, dimBlock>>>();
    timer.stop();

    if (cudaGetLastError() != cudaSuccess)
    {
        printf("kernel error evaluateTileRanges\n");
    }
}

__global__ void rasterizeTilesKernel()
{
    __shared__ int32_t s_FirstSplatIndex;
    __shared__ int32_t s_SplatsCount;
    __shared__ float4 s_Colors[k_WarpSize];
    __shared__ float4 s_Conics[k_WarpSize];
    __shared__ float2 s_Centers[k_WarpSize];

    if (threadIdx.x == 0)
    {
        s_FirstSplatIndex = loadReadOnly(&g_GlobalArgs.tileRange[blockIdx.x * 2]);
        auto endSplatIndex = loadReadOnly(&g_GlobalArgs.tileRange[blockIdx.x * 2 + 1]);
        s_SplatsCount = endSplatIndex - s_FirstSplatIndex;
    }

    __syncthreads();

    if (s_SplatsCount == 0)
    {
        return;
    }

    const auto threadTilePixelCoords = glm::ivec2(threadIdx.x % k_TileSize, threadIdx.x / k_TileSize);
    const auto threadBufferPixelCoords =
        glm::ivec2(blockIdx.x % k_TilesPerScreen, blockIdx.x / k_TilesPerScreen) * k_TileSize + threadTilePixelCoords;
    const auto clipCoords = (glm::vec2) threadBufferPixelCoords * (2.0f / (float) k_ScreenSize) - 1.0f;

    auto color = glm::vec3(0.0f);
    auto transmittance = 1.0f;

    while (s_SplatsCount > 0)
    {
        auto splatsCount = glm::min(s_SplatsCount, k_WarpSize);

        __syncthreads();

        if (threadIdx.x < splatsCount)
        {
            auto srcIndex = g_TileListArgs.values[s_FirstSplatIndex + threadIdx.x];
            s_Centers[threadIdx.x] = loadReadOnly(&g_GlobalArgs.positionClipSpaceXY[srcIndex]);
            auto conicData = loadReadOnly(&g_GlobalArgs.conic[srcIndex]);
            auto colorData = loadReadOnly(&g_GlobalArgs.color[srcIndex]);
            s_Colors[threadIdx.x] = float4{colorData.x, colorData.y, colorData.z, colorData.w};
            s_Conics[threadIdx.x] = float4{conicData.x, conicData.y, conicData.z, 0.0f};
        }

        __syncthreads();

        for (auto i = 0; i != splatsCount; ++i)
        {
            auto d = clipCoords - builtinToGlmVec2(s_Centers[i]);
            auto splatColor = builtinToGlmVec4(s_Colors[i]);
            auto conic = s_Conics[i];
            auto dx = conic.x * d.x * d.x + conic.z * d.y * d.y + 2.0f * conic.y * d.x * d.y;
            auto density = __expf(-0.5f * dx);
            auto alpha = splatColor.a * __saturatef(density);
            color += splatColor.rgb * transmittance * alpha;
            transmittance *= (1.0f - alpha);
        }

        if (threadIdx.x == 0)
        {
            s_FirstSplatIndex += splatsCount;
            s_SplatsCount -= splatsCount;
        }

        if (__syncthreads_count(transmittance > 0.02f) == 0)
        {
            break;
        }
    }

    uchar4 quantizedColor;
    quantizedColor.x = color.x * 255;
    quantizedColor.y = color.y * 255;
    quantizedColor.z = color.z * 255;
    quantizedColor.w = 255;

    auto globalWriteIndex = threadBufferPixelCoords.y * k_ScreenSize + threadBufferPixelCoords.x;
    g_GlobalArgs.backBuffer[globalWriteIndex] = quantizedColor;
}

void rasterizeTile(CudaTimer& timer)
{
    constexpr int32_t threadPerBlock{k_TileSize * k_TileSize};
    const auto dimBlock{dim3(threadPerBlock)};
    const auto dimGrid{dim3(k_TotalTiles)};

    timer.start();
    rasterizeTilesKernel<<<dimGrid, dimBlock>>>();
    timer.stop();

    if (cudaGetLastError() != cudaSuccess)
    {
        printf("kernel error renderDepthBuffer\n");
    }
}
