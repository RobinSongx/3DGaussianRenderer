#include "ply_loader.h"
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <map>
#include <array>
#include <stdexcept>
#include <limits>
#include <glm/glm.hpp>

namespace gauss_render {

// Sigmoid激活函数
// 将logit转换到[0, 1]范围，用于不透明度
float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// 解析PLY文件头
// 读取属性列表和顶点数量
void parseHeader(std::ifstream& inFile, std::vector<std::string>& properties, int& vertexCount) {
    vertexCount = -1;

    std::string line;
    std::string placeholder;

    // 防止坏文件死循环，限制最大行数
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

// 将vec4编码到uint32（每个分量8位）
// 用于压缩四元数，每个分量范围在[0, 1]
uint32_t encodeVec4(const glm::vec4& v) {
    glm::vec4 clamped = glm::clamp(v, 0.0f, 1.0f);
    // clang-format off
    return (
        ((uint32_t) (clamped.x * 255.0f) << 24u) | 
        ((uint32_t) (clamped.y * 255.0f) << 16u) | 
        ((uint32_t) (clamped.z * 255.0f) <<  8u ) | 
        ((uint32_t) (clamped.w * 255.0f)       ));
    // clang-format on
}

// 获取属性在properties数组中的索引
// 如果找不到抛出异常
int indexOf(const std::vector<std::string>& properties, const std::string& property) {
    auto it = std::find(properties.begin(), properties.end(), property);
    if (it == properties.end()) {
        throw std::invalid_argument("Missing property \"" + property + "\".");
    }
    return std::distance(properties.begin(), it);
}

// 从PLY文件加载高斯模型
int LoadPly(const std::string& path, GaussianModel& model) {
    try {
        std::vector<std::string> properties;
        std::vector<std::vector<float>> propertiesData;
        int vertexCount;

        // 打开二进制文件
        std::ifstream inFile(path, std::ios_base::binary);
        if (!inFile.is_open()) {
            printf("Failed to open file: %s\n", path.c_str());
            return 0;
        }

        // 解析文件头，获取属性列表和顶点数量
        parseHeader(inFile, properties, vertexCount);

        const auto numProps = properties.size();
        propertiesData.resize(numProps);
        for (int i = 0; i != numProps; ++i) {
            propertiesData[i].resize(vertexCount);
        }

        // 读取所有顶点的属性数据
        for (int i = 0; i != vertexCount; ++i) {
            for (int j = 0; j != numProps; ++j) {
                float val;
                inFile.read(reinterpret_cast<char*>(&val), sizeof(float));
                propertiesData[j][i] = val;
            }
        }

        // 获取各个必填属性在properties数组中的索引
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

        // 统计额外的球面谐波系数（f_rest_*）
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

        // 根据额外系数数量推断球谐最大阶数
        // 阶数d的总系数 = (d+1)^2 * 3，减去3个0阶就是额外系数数量
        int shDegree = 0;
        int expectedShCount = 0;
        while (expectedShCount < extraShCount) {
            expectedShCount += (2 * (shDegree + 1) + 1) * 3;
            ++shDegree;
        }

        int totalShCount = extraShCount + 3;
        model.sh_degree = shDegree;

        // 分配内存存储所有高斯
        model.num_gaussians = vertexCount;
        model.gaussians.resize(vertexCount);
        model.sh_coefficients.resize(vertexCount * totalShCount);

        // 初始化边界框
        glm::vec3 boundsMin(std::numeric_limits<float>::max());
        glm::vec3 boundsMax(std::numeric_limits<float>::min());

        // 逐个解析每个高斯
        for (int i = 0; i != vertexCount; ++i) {
            // 读取世界空间位置
            glm::vec3 translation(
                propertiesData[offsets[0]][i],
                propertiesData[offsets[1]][i],
                propertiesData[offsets[2]][i]
            );

            // 读取旋转四元数
            glm::quat rotation(
                propertiesData[offsets[3]][i],
                propertiesData[offsets[4]][i],
                propertiesData[offsets[5]][i],
                propertiesData[offsets[6]][i]
            );

            // 读取缩放因子（对数空间）
            glm::vec3 scale(
                propertiesData[offsets[7]][i],
                propertiesData[offsets[8]][i],
                propertiesData[offsets[9]][i]
            );

            // 读取0阶球谐颜色系数
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
            // 不透明度从logit转换到[0, 1]
            float opacity = sigmoid(opacity_logit);

            // 更新场景包围盒
            boundsMin = glm::min(boundsMin, translation);
            boundsMax = glm::max(boundsMax, translation);

            // 球谐0阶系数转换到[0, 1]范围
            const float SH_C0 = 0.28209479177387814f;
            rgb = rgb * SH_C0 + 0.5f;

            // 填充高斯结构
            Gaussian g;
            g.position[0] = translation.x;
            g.position[1] = translation.y;
            g.position[2] = translation.z;
            g.opacity = opacity;
            g.scale[0] = scale.x;
            g.scale[1] = scale.y;
            g.scale[2] = scale.z;
            g.rotation[0] = rotation.w;
            g.rotation[1] = rotation.x;
            g.rotation[2] = rotation.y;
            g.rotation[3] = rotation.z;
            g.color[0] = rgb.r;
            g.color[1] = rgb.g;
            g.color[2] = rgb.b;

            model.gaussians[i] = g;

            // 拷贝所有球谐系数到输出
            int shIdx = totalShCount * i + 3;
            model.sh_coefficients[shIdx + 0] = propertiesData[offsets[10]][i];
            model.sh_coefficients[shIdx + 1] = propertiesData[offsets[11]][i];
            model.sh_coefficients[shIdx + 2] = propertiesData[offsets[12]][i];
            for (int j = 0; j != extraShCount; ++j) {
                model.sh_coefficients[shIdx + 3 + j] = propertiesData[shOffsets[j]][i];
            }
        }

        // 保存场景包围盒
        model.bounds_min = boundsMin;
        model.bounds_max = boundsMax;

        // 预处理球谐系数（重排列以适应GPU合并访问）
        model.PreprocessSh();

        printf("Loaded %d gaussians from %s\n", vertexCount, path.c_str());
        return vertexCount;
    } catch (const std::invalid_argument& e) {
        printf("Failed to parse PLY: %s\n", e.what());
        return 0;
    }
}

} // namespace gauss_render
