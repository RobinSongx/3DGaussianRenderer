#define GLM_FORCE_SWIZZLE

#include <string>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>
GLFWwindow* window;

#include "../core/consts.h"
#include "cuda_gl_interop.h"
#include "../core/device_buffer.cu"
#include "../core/gaussian_render.cuh"
#include "../core/ply_parser.h"
#include "../core/camera_controls.h"

#define STRINGIFY(A) #A

const char* vertexShaderTextured = STRINGIFY(
    #version 330 core \n
    layout(location = 0) in vec3 vertexPosition_modelspace; \n 
    layout(location = 1) in vec2 vertexUv; \n 
    out vec2 uv; \n 
    void main() \n 
    { \n 
        gl_Position = vec4(vertexPosition_modelspace, 1); \n
        uv = vertexUv; \n
    } \n
);

const char* fragmentShaderTextured = STRINGIFY(
    #version 330 core \n
    in vec2 uv; \n 
    out vec4 color; \n 
    uniform sampler2D textureSampler; \n
    void main() \n 
    { \n 
        color = texture(textureSampler, uv).rgba; \n
    } \n
);

GLuint compileShadersProgram(const char* vertexShader, const char* fragmentShader)
{
    auto vertexShaderId = glCreateShader(GL_VERTEX_SHADER);
    auto fragmentShaderId = glCreateShader(GL_FRAGMENT_SHADER);

    GLint result = GL_FALSE;
    int infoLogLength;

    glShaderSource(vertexShaderId, 1, &vertexShader, NULL);
    glCompileShader(vertexShaderId);

    glGetShaderiv(vertexShaderId, GL_COMPILE_STATUS, &result);
    glGetShaderiv(vertexShaderId, GL_INFO_LOG_LENGTH, &infoLogLength);

    if (infoLogLength > 0)
    {
        std::vector<char> VertexShaderErrorMessage(infoLogLength + 1);
        glGetShaderInfoLog(vertexShaderId, infoLogLength, NULL, &VertexShaderErrorMessage[0]);
        printf("%s\n", &VertexShaderErrorMessage[0]);
    }

    glShaderSource(fragmentShaderId, 1, &fragmentShader, NULL);
    glCompileShader(fragmentShaderId);

    glGetShaderiv(fragmentShaderId, GL_COMPILE_STATUS, &result);
    glGetShaderiv(fragmentShaderId, GL_INFO_LOG_LENGTH, &infoLogLength);

    if (infoLogLength > 0)
    {
        std::vector<char> FragmentShaderErrorMessage(infoLogLength + 1);
        glGetShaderInfoLog(fragmentShaderId, infoLogLength, NULL, &FragmentShaderErrorMessage[0]);
        printf("%s\n", &FragmentShaderErrorMessage[0]);
    }

    auto programId = glCreateProgram();
    glAttachShader(programId, vertexShaderId);
    glAttachShader(programId, fragmentShaderId);
    glLinkProgram(programId);

    glGetProgramiv(programId, GL_LINK_STATUS, &result);
    glGetProgramiv(programId, GL_INFO_LOG_LENGTH, &infoLogLength);

    if (infoLogLength > 0)
    {
        std::vector<char> ProgramErrorMessage(infoLogLength + 1);
        glGetProgramInfoLog(programId, infoLogLength, NULL, &ProgramErrorMessage[0]);
        printf("%s\n", &ProgramErrorMessage[0]);
    }

    glDetachShader(programId, vertexShaderId);
    glDetachShader(programId, fragmentShaderId);

    glDeleteShader(vertexShaderId);
    glDeleteShader(fragmentShaderId);

    return programId;
}

float4 glmToBuiltin4(const glm::vec4& v)
{
    return float4{v.x, v.y, v.z, v.w};
}

void generateRandomGaussians(
    std::vector<float4>& position,
    std::vector<float4>& scaleAndRotation,
    std::vector<float4>& color,
    const float minScale,
    const float maxScale,
    const glm::vec4& minPosition,
    const glm::vec4& maxPosition)
{
    for (auto i = 0; i != position.size(); ++i)
    {
        auto translation = glm::linearRand(minPosition, maxPosition);
        auto rotAxis = glm::sphericalRand(1.0f);
        auto rotAngle = glm::linearRand(0.0f, glm::pi<float>());
        auto rotation = glm::angleAxis(rotAngle, rotAxis);
        auto scale = glm::linearRand(glm::vec3(minScale, minScale, minScale), glm::vec3(maxScale, maxScale, maxScale));

        auto col = glm::linearRand(glm::vec4(0), glm::vec4(1));
        auto quantizedRotation = encodeVec4((glm::vec4(rotation.x, rotation.y, rotation.z, rotation.w) + 1.0f) * 0.5f);

        position[i] = float4{translation.x, translation.y, translation.z, 1.0f};
        scaleAndRotation[i] = float4{scale.x, scale.y, scale.z, reinterpret_cast<float&>(quantizedRotation)};
        color[i] = float4{col.x, col.y, col.z, col.w};
    }
}

constexpr uint32_t k_QuadIndices[] = {0, 1, 2, 2, 3, 0};
const glm::vec3 k_QuadVertices[] = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};
const glm::vec2 k_QuadUvs[] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};

struct Stats
{
    double evaluateSphericalHarmonics{0};
    double evaluateClipData{0};
    double buildTileList{0};
    double sortTileList{0};
    double evaluateTileRanges{0};
    double renderDepthBuffer{0};
};

void realignSphericalHarmonics(
    const std::vector<float>& srcSh,
    std::vector<float>& dstSh,
    const int groupSize,
    const int shCount,
    const int splatCount)
{
    auto shCountPerComponent = shCount / 3;
    assert(shCountPerComponent * 3 == shCount);

    auto idx = 0;
    auto groupCount = (int) glm::ceil(splatCount / (float) groupSize);

    dstSh.resize(groupCount * groupSize * shCount);

    for (auto grp = 0; grp != groupCount; ++grp)
    {
        auto start = grp * groupSize * shCount;
        auto thisGroupSize = glm::min(groupSize, splatCount - grp * groupSize);

        for (auto i = 0; i != shCount; ++i)
        {
            for (auto k = 0; k != thisGroupSize; ++k)
            {
                auto srcIdx = shCount * k + i;
                auto dstIdx = groupSize * i + k;
                dstSh[start + dstIdx] = srcSh[start + srcIdx];
            }
        }
    }
}

int main(int argc, char* argv[])
{
    if (!glfwInit())
    {
        fprintf(stderr, "Failed to initialize GLFW\n");
        getchar();
        return -1;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(k_ScreenSize, k_ScreenSize, "3DGaussianRenderer", NULL, NULL);
    if (window == NULL)
    {
        fprintf(stderr,
                "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 "
                "compatible. Try the 2.1 version of the tutorials.\n");
        getchar();
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    glewExperimental = true;
    if (glewInit() != GLEW_OK)
    {
        fprintf(stderr, "Failed to initialize GLEW\n");
        getchar();
        glfwTerminate();
        return -1;
    }

    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    cudaDeviceProp prop;
    int dev;
    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 0;
    checkCudaErrors(cudaChooseDevice(&dev, &prop));
    checkCudaErrors(cudaGLSetGLDevice(dev));

    cudaGetDeviceProperties(&prop, dev);

    auto cameraControls = CameraControls(window, glm::vec2(k_ScreenSize));

    std::vector<float> sphericalHarmonics;
    int sphericalHarmonicsDegree{0};
    int sphericalHarmonicsCount{0};

#if false
    constexpr int32_t splatCount = 1 << 16;
    constexpr auto worldBoundsExtent = 4.0f;

    const auto minPosition = glm::vec4(-worldBoundsExtent, -worldBoundsExtent, -worldBoundsExtent, 1.0f);
    const auto maxPosition = glm::vec4(worldBoundsExtent, worldBoundsExtent, worldBoundsExtent, 1.0f);

    auto position = std::vector<float4>(splatCount);
    auto scaleAndRotation = std::vector<float4>(splatCount);
    auto color = std::vector<float4>(splatCount);
    generateRandomGaussians(position, scaleAndRotation, color, 0.01f, 0.2f, minPosition, maxPosition);

    if (savePly("random_point_cloud.ply", position, scaleAndRotation, color))
    {
        printf("Random point cloud saved to random_point_cloud.ply\n");
    }
    else
    {
        printf("Failed to save random point cloud\n");
    }

    cameraControls.setBounds(glm::vec3(-worldBoundsExtent), glm::vec3(worldBoundsExtent));

#else
    std::vector<float4> position;
    std::vector<float4> scaleAndRotation;
    std::vector<float4> color;
    glm::vec3 boundsMin;
    glm::vec3 boundsMax;
    auto splatCount = parsePly(
        "d:\\code\\deep_Learning\\3DGaussianRenderer\\data\\random_cube.ply",
        position,
        scaleAndRotation,
        color,
        sphericalHarmonics,
        sphericalHarmonicsDegree,
        sphericalHarmonicsCount,
        boundsMin,
        boundsMax);
    auto hasExtraSphericalHarmonics = sphericalHarmonicsDegree > 1;

    std::vector<float> alignedSphericalHarmonics;
    if (sphericalHarmonicsDegree != 0)
    {
        realignSphericalHarmonics(
            sphericalHarmonics, alignedSphericalHarmonics, 256, sphericalHarmonicsCount, splatCount);
        std::swap(sphericalHarmonics, alignedSphericalHarmonics);
    }

    cameraControls.setBounds(boundsMin, boundsMax);

#endif

    auto programTexturedId = compileShadersProgram(vertexShaderTextured, fragmentShaderTextured);
    auto textureId = glGetUniformLocation(programTexturedId, "textureSampler");

    auto vertexArray = GLVertexArray();
    auto glVertexBuffer = GLBuffer<glm::vec3, GL_ARRAY_BUFFER>(4, k_QuadVertices);
    auto glUvBuffer = GLBuffer<glm::vec2, GL_ARRAY_BUFFER>(4, k_QuadUvs);
    auto glIndexBuffer = GLBuffer<uint32_t, GL_ARRAY_BUFFER>(6, k_QuadIndices);
    auto glColorBuffer = GLBuffer<uchar4, GL_PIXEL_UNPACK_BUFFER>(k_ScreenSize * k_ScreenSize);

    auto positionBuffer = DeviceBuffer<float4>(position);
    auto scaleAndRotationBuffer = DeviceBuffer<float4>(scaleAndRotation);
    auto colorBuffer = DeviceBuffer<float4>(color);
    auto sphericalHarmonicsBuffer = DeviceBuffer<float>(sphericalHarmonics);
    auto conicAndColorBuffer = DeviceBuffer<float4>(splatCount);
    auto positionClipSpaceXYBuffer = DeviceBuffer<float2>(splatCount);
    auto positionClipSpaceZBuffer = DeviceBuffer<float>(splatCount);
    auto screenEllipseBuffer = DeviceBuffer<float4>(splatCount);
    auto tileRangeBuffer = DeviceBuffer<int32_t>(k_TotalTiles * 2);

    auto tileListCapacity = splatCount * 8;

    auto tileListKeysCurrentBuffer = DeviceBuffer<uint64_t>(tileListCapacity);
    auto tileListValuesCurrentBuffer = DeviceBuffer<int32_t>(tileListCapacity);
    auto tileListKeysAlternateBuffer = DeviceBuffer<uint64_t>(tileListCapacity);
    auto tileListValuesAlternateBuffer = DeviceBuffer<int32_t>(tileListCapacity);

    void* deviceTempStorage{nullptr};
    size_t tempStorageSizeInBytes{0};

    auto colorBufferResource = CudaGraphicsResource(glColorBuffer.getBufferId(), cudaGraphicsRegisterFlagsNone);

    Stats stats;
    long frameCount{0};
    auto cudaTimer = CudaTimer();

    auto lastTime = glfwGetTime();
    auto deltaTime{0.0f};

    auto tileListIsSaturated{false};

    do
    {
        ++frameCount;

        if (tileListIsSaturated)
        {
            tileListCapacity <<= 1;
            tileListKeysCurrentBuffer.resizeIfNeeded(tileListCapacity);
            tileListValuesCurrentBuffer.resizeIfNeeded(tileListCapacity);
            tileListKeysAlternateBuffer.resizeIfNeeded(tileListCapacity);
            tileListValuesAlternateBuffer.resizeIfNeeded(tileListCapacity);
            tileListIsSaturated = false;
        }

        auto time = glfwGetTime();
        deltaTime = time - lastTime;
        lastTime = time;

        cameraControls.update((float) deltaTime);

        CameraData cameraData;
        cameraData.position = cameraControls.getPosition();
        cameraData.aspect = cameraControls.getAspect();
        cameraData.projection = cameraControls.getProjection();
        cameraData.viewProjection = cameraControls.getViewProjection();
        cameraData.view = cameraControls.getView();
        auto cotangentY = 1.0f / glm::tan(cameraControls.getFieldOfView() * 0.5f);
        auto cotangentX = cotangentY / cameraControls.getAspect();
        cameraData.fovCotangent = glm::vec2(cotangentX, cotangentY);
        auto scaleZ = -2.0f / (cameraControls.getFar() - cameraControls.getNear());
        auto translationZ = -(cameraControls.getFar() + cameraControls.getNear())
                          / (cameraControls.getFar() - cameraControls.getNear());
        cameraData.depthScaleBias = glm::vec2(scaleZ, translationZ);

        {
            auto colorBinding = colorBufferResource.getBinding<uchar4>();

            checkCudaErrors(cudaMemset(colorBinding.getPtr(), 0, glColorBuffer.getSizeInBytes()));

            tileRangeBuffer.clearMemory(255);

            GlobalArgs globalArgs;
            globalArgs.splatCount = splatCount;
            globalArgs.positionClipSpaceXY = positionClipSpaceXYBuffer.getPtr();
            globalArgs.positionClipSpaceZ = positionClipSpaceZBuffer.getPtr();
            globalArgs.sphericalHarmonicsDegree = sphericalHarmonicsDegree;
            globalArgs.sphericalHarmonicsCount = sphericalHarmonicsCount;
            globalArgs.sphericalHarmonics = sphericalHarmonicsBuffer.getPtr();
            globalArgs.screenEllipse = screenEllipseBuffer.getPtr();
            globalArgs.position = positionBuffer.getPtr();
            globalArgs.scaleAndRotation = scaleAndRotationBuffer.getPtr();
            globalArgs.color = colorBuffer.getPtr();
            globalArgs.conic = conicAndColorBuffer.getPtr();
            globalArgs.tileRange = tileRangeBuffer.getPtr();
            globalArgs.backBuffer = colorBinding.getPtr();
            globalArgs.cameraData = cameraData;

            setGlobalArgs(&globalArgs);

            TileListArgs tileListArgs;
            tileListArgs.keys = tileListKeysCurrentBuffer.getPtr();
            tileListArgs.values = tileListValuesCurrentBuffer.getPtr();
            tileListArgs.capacity = tileListCapacity;

            setTileListArgs(&tileListArgs);

            if (sphericalHarmonicsDegree != 0)
            {
                evaluateSphericalHarmonics(cudaTimer, splatCount);
                stats.evaluateSphericalHarmonics += (double) cudaTimer.getElapseTimedMs();
            }

            evaluateSplatClipData(cudaTimer, splatCount);
            stats.evaluateClipData += (double) cudaTimer.getElapseTimedMs();

            tileListArgs.size = buildTileList(cudaTimer, prop.multiProcessorCount * 8, tileListCapacity);
            assert(tileListArgs.size <= tileListCapacity);
            stats.buildTileList += (double) cudaTimer.getElapseTimedMs();

            tileListIsSaturated = tileListArgs.size == tileListCapacity;

            setTileListArgs(&tileListArgs);

            DoubleBuffer<uint64_t> tileListKeys(
                tileListKeysCurrentBuffer.getPtr(), tileListKeysAlternateBuffer.getPtr());
            DoubleBuffer<int32_t> tileListValues(
                tileListValuesCurrentBuffer.getPtr(), tileListValuesAlternateBuffer.getPtr());

            sortTileList(
                cudaTimer, tileListArgs.size, deviceTempStorage, tempStorageSizeInBytes, tileListKeys, tileListValues);
            stats.sortTileList += (double) cudaTimer.getElapseTimedMs();

            tileListArgs.keys = tileListKeys.current();
            tileListArgs.values = tileListValues.current();

            setTileListArgs(&tileListArgs);

            if (tileListArgs.size != 0)
            {
                evaluateTileRange(cudaTimer, tileListArgs.size);
                stats.evaluateTileRanges += (double) cudaTimer.getElapseTimedMs();
            }

            rasterizeTile(cudaTimer);
            stats.renderDepthBuffer += (double) cudaTimer.getElapseTimedMs();

            checkCudaErrors(cudaDeviceSynchronize());
        }

        glClear(GL_COLOR_BUFFER_BIT);
        glDisable(GL_DEPTH_TEST);

        {
            glUseProgram(programTexturedId);
            glEnable(GL_TEXTURE_2D);

            auto vertexBinding = glVertexBuffer.getBinding(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*) 0);
            auto uvBinding = glUvBuffer.getBinding(1);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void*) 0);

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, 0);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            auto depthBinding = glColorBuffer.getBinding();
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, k_ScreenSize, k_ScreenSize, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

            glUniform1i(textureId, 0);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glIndexBuffer.getBufferId());
            glDrawElements(GL_TRIANGLES, glIndexBuffer.getSize(), GL_UNSIGNED_INT, (void*) 0);

            glDisable(GL_TEXTURE_2D);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();

        while (glfwGetTime() < lastTime + 1.0 / 60.0)
        {
        }

    } while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

    if (deviceTempStorage != nullptr)
    {
        checkCudaErrors(cudaFree(deviceTempStorage));
    }

    glDeleteProgram(programTexturedId);

    glfwTerminate();

    stats.evaluateSphericalHarmonics /= (double) frameCount;
    stats.evaluateClipData /= (double) frameCount;
    stats.buildTileList /= (double) frameCount;
    stats.sortTileList /= (double) frameCount;
    stats.evaluateTileRanges /= (double) frameCount;
    stats.renderDepthBuffer /= (double) frameCount;

    auto totalMs =
        stats.evaluateClipData +
        stats.buildTileList +
        stats.sortTileList +
        stats.evaluateTileRanges +
        stats.renderDepthBuffer;

    printf("evaluateSphericalHarmonics average time ms: %2.6f\n", stats.evaluateSphericalHarmonics);
    printf("evaluateClipData average time ms: %2.6f\n", stats.evaluateClipData);
    printf("buildTileList average time ms: %2.6f\n", stats.buildTileList);
    printf("sortTileList average time ms: %2.6f\n", stats.sortTileList);
    printf("evaluateTileRanges average time ms: %2.6f\n", stats.evaluateTileRanges);
    printf("renderDepthBuffer average time ms: %2.6f\n", stats.renderDepthBuffer);
    printf("Total average time ms: %2.6f\n", totalMs);
    getchar();

    return 0;
}
