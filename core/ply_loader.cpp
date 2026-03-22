#include "ply_loader.h"
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <map>
#include <array>
#include <stdexcept>
#include <glm/glm.hpp>

namespace gauss_render {

// sigmoid激活函数
float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// 解析PLY文件头
void parseHeader(std::ifstream& inFile, std::vector<std::string>& properties, int& vertexCount) {
    vertexCount = -1;

    std::string line;
    std::string placeholder;

    // 防止坏文件死循环
    constexpr int maxLength = 256;
    int iteration = 0;

    while (std::getline(inFile, line)) {
        std::stringstream ss(line);
        std::string word;

        if (!(ss >> word)) {
            continue;
        }

        if (word == "ply") {
            continue;
        } else if (word == "format") {
            // 只支持binary_little_endian
            continue;
        } else if (word == "element") {
            if (!(ss >> word && word == "vertex")) {
                throw std::invalid_argument("Unexpected element type. Got: " + word);
            }

            if (!(ss >> word)) {
                throw std::invalid_argument("Unexpected element count.");
            }

            vertexCount = std::stoi(word);
        } else if (word == "property") {
            if (!(ss >> word && word == "float")) {
                throw std::invalid_argument("Unexpected property format, expected float. Got: " + word);
            }

            if (!(ss >> word)) {
                throw std::invalid_argument("Unexpected property, missing name.");
            }

            if (std::find(properties.begin(), properties.end(), word) != properties.end()) {
                throw std::invalid_argument("Duplicated property \"" + word + "\".");
            }

            properties.push_back(word);
        } else if (word == "end_header") {
            break;
        }

        ++iteration;

        if (iteration > maxLength) {
            throw std::invalid_argument("Invalid header. End not found after " + std::to_string(maxLength) + " lines.");
        }
    }
}

// 编码四元数到uint32
uint32_t encodeVec4(const glm::vec4& v) {
    glm::vec4 clamped = glm::clamp(v);
    // clang-format off
    return (
        ((uint32_t) (clamped.x * 255.0f) << 24u) | 
        ((uint32_t) (clamped.y * 255.0f) << 16u) | 
        ((uint32_t) (clamped.z * 255.0f) <<  8u) | 
        ((uint32_t) (clamped.w * 255.0f)       ));
    // clang-format on
}

// 获取属性在properties数组中的索引
int indexOf(const std::vector<std::string>& properties, const std::string& property) {
    auto it = std::find(properties.begin(), properties.end(), property);
    if (it == properties.end()) {
        throw std::invalid_argument("Missing property \"" + property + "\".");
    }
    return std::distance(properties.begin(), it);
}

// 加载PLY文件
int LoadPly(const std::string& path, GaussianModel& model) {
    try {
        std::vector<std::string> properties;
        std::vector<std::vector<float>> propertiesData;
        int vertexCount;

        // 打开文件
        std::ifstream inFile(path, std::ios_base::binary);
        if (!inFile.is_open()) {
            printf("Failed to open file: %s\n", path.c_str());
            return 0;
        }

        // 解析头部
        parseHeader(inFile, properties, vertexCount);

        const auto numProps = properties.size();
        propertiesData.resize(numProps);
        for (int i = 0; i != numProps; ++i) {
            propertiesData[i].resize(vertexCount);
        }

        // 读取顶点数据
        for (int i = 0; i != vertexCount; ++i) {
            for (int j = 0; j != numProps; ++j) {
                float val;
                inFile.read(reinterpret_cast<char*>(&val), sizeof(float));
                propertiesData[j][i] = val;
            }
        }

        // 获取各个属性的索引
        std::array<int, 14> offsets{};
        offsets[0] = indexOf(properties, "x");
        offsets[1] = indexOf(properties, "y");
        offsets[2] = indexOf(properties, "z");
        offsets[3] = indexOf(properties, "rot_0");
        offsets[4] = indexOf(properties, "rot_1");
        offsets[5] = indexOf(properties, "rot_2");
        offsets[6] = indexOf(properties, "rot_3");
        offsets[7] = indexOf(properties, "scale_0");
        offsets[8] = indexOf(properties, "scale_1");
        offsets[9] = indexOf(properties, "scale_2");
        offsets[10] = indexOf(properties, "f_dc_0");
        offsets[11] = indexOf(properties, "f_dc_1");
        offsets[12] = indexOf(properties, "f_dc_2");
        offsets[13] = indexOf(properties, "opacity");

        // 处理额外的球面谐波系数
        int extraShCount = 0;
        std::vector<int> shOffsets;
        for (;;) {
            auto it = std::find(properties.begin(), properties.end(), "f_rest_" + std::to_string(extraShCount));
            if (it == properties.end()) {
                break;
            } else {
                shOffsets.push_back(std::distance(properties.begin(), it));
            }
            ++extraShCount;
        }

        // 计算球面谐波度数
        int shDegree = 0;
        int expectedShCount = 0;
        while (expectedShCount < extraShCount) {
            expectedShCount += (2 * (shDegree + 1) + 1) * 3;
            ++shDegree;
        }

        int totalShCount = extraShCount + 3;
        model.max_sh_degree = shDegree;

        // 分配空间
        model.num_gaussians = vertexCount;
        model.positions.resize(vertexCount);
        model.scale_rotation.resize(vertexCount);
        model.colors.resize(vertexCount);
        model.shs.resize(vertexCount * totalShCount);

        // 计算边界框
        glm::vec3 boundsMin(std::numeric_limits<float>::max());
        glm::vec3 boundsMax(std::numeric_limits<float>::min());

        // 解析每个顶点
        for (int i = 0; i != vertexCount; ++i) {
            // 读取位置
            glm::vec3 translation(
                propertiesData[offsets[0]][i],
                propertiesData[offsets[1]][i],
                propertiesData[offsets[2]][i]
            );

            // 读取旋转四元数
            glm::quat rotation = glm::quat::wxyz(
                propertiesData[offsets[3]][i],
                propertiesData[offsets[4]][i],
                propertiesData[offsets[5]][i],
                propertiesData[offsets[6]][i]
            );

            // 读取缩放
            glm::vec3 scale(
                propertiesData[offsets[7]][i],
                propertiesData[offsets[8]][i],
                propertiesData[offsets[9]][i]
            );

            // 读取基色（球面谐波0阶）
            glm::vec3 rgb(
                propertiesData[offsets[10]][i],
                propertiesData[offsets[11]][i],
                propertiesData[offsets[12]][i]
            );

            // 读取不透明度logit
            float opacity_logit = propertiesData[offsets[13]][i];

            // 归一化旋转四元数
            rotation = glm::normalize(rotation);
            // 缩放从对数空间转换到线性空间
            scale = glm::exp(scale);
            // 不透明度从logit转换到[0,1]
            float opacity = sigmoid(opacity_logit);

            // 更新边界框
            boundsMin = glm::min(boundsMin, translation);
            boundsMax = glm::max(boundsMax, translation);

            // RGB颜色 球面谐波 → [0,1]
            const float SH_C0 = 0.28209479177387814f;
            rgb = rgb * SH_C0 + 0.5f;

            // 编码旋转四元数到uint32（四个8位分量）
            uint32_t quantizedRot = encodeVec4((rotation + 1.0f) * 0.5f);

            // 打包位置 + 不透明度
            model.positions[i] = float4{translation.x, translation.y, translation.z, opacity};
            // 打包缩放 + 压缩旋转
            model.scale_rotation[i] = float4{scale.x, scale.y, scale.z, reinterpret_cast<float&>(quantizedRot)};
            // 打包RGB颜色 + 不透明度
            model.colors[i] = float4{rgb.r, rgb.g, rgb.b, opacity};

            // 拷贝球面谐波系数到模型
            int shIdx = totalShCount * i + 3;
            for (int j = 0; j != 3; ++j) {
                model.shs[shIdx + j] = propertiesData[offsets[10 + j]][i];
            }
            for (int j = 0; j != extraShCount; ++j) {
                model.shs[shIdx + 3 + j] = propertiesData[shOffsets[j]][i];
            }
        }

        // 保存边界框
        model.bounds_min = boundsMin;
        model.bounds_max = boundsMax;

        // 预处理球面谐波（暂存，实际计算在上传到GPU时调用）
        model.PreprocessSh();

        printf("Loaded %d gaussians from %s\n", vertexCount, path.c_str());
        return vertexCount;
    } catch (const std::invalid_argument& e) {
        printf("Failed to parse PLY: %s\n", e.what());
        return 0;
    }
}

} // namespace gauss_render
