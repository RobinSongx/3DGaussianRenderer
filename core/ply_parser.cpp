#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <map>
#include <array>
#include <stdexcept>
#include "ply_parser.h"

float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

void parseHeader(std::ifstream& inFile, std::vector<std::string>& properties, int& vertexCount)
{
    vertexCount = -1;

    std::string lineStr;
    std::string placeholder;

    std::string line;
    auto littleEndian{false};

    constexpr int maxLength{256};
    auto iteration{0};

    while (std::getline(inFile, line))
    {
        std::stringstream ss(line);
        std::string word;

        if (!(ss >> word))
        {
            throw std::invalid_argument("Unexpected empty line.");
        }

        if (word == "ply")
        {
            continue;
        }

        else if (word == "format")
        {
            littleEndian = ss >> word && word == "binary_little_endian";
        }

        else if (word == "element")
        {
            if (!(ss >> word && word == "vertex"))
            {
                throw std::invalid_argument("Unexpected element type.");
            }

            if (!(ss >> word))
            {
                throw std::invalid_argument("Unexpected element count.");
            }

            vertexCount = std::stoi(word);
        }

        else if (word == "property")
        {
            if (!(ss >> word && word == "float"))
            {
                throw std::invalid_argument("Unexpected property format, expected float.");
            }

            if (!(ss >> word))
            {
                throw std::invalid_argument("Unexpected property, missing name.");
            }

            if (std::find(properties.begin(), properties.end(), word) != properties.end())
            {
                throw std::invalid_argument("Duplicated property \"" + word + "\".");
            }

            properties.push_back(word);
        }

        else if (word == "end_header")
        {
            if (!littleEndian || vertexCount == -1)
            {
                throw std::invalid_argument("Invalid header. Format or vertex count not found.");
            }

            break;
        }

        ++iteration;

        if (iteration > maxLength)
        {
            throw std::invalid_argument("Invalid header. End not found.");
        }
    }
}

int parsePly(const char* filePath, std::vector<std::string>& properties, std::vector<std::vector<float>>& propertiesData)
{
    assert(properties.size() == 0);
    assert(propertiesData.size() == 0);

    std::ifstream inFile(filePath, std::ios_base::binary);

    auto vertexCount{-1};
    parseHeader(inFile, properties, vertexCount);

    const auto numProps = properties.size();
    propertiesData.resize(numProps);
    for (auto i = 0; i != numProps; ++i)
    {
        propertiesData[i] = std::vector<float>();
        propertiesData[i].resize(vertexCount);
    }

    for (auto i = 0; i != vertexCount; ++i)
    {
        for (auto j = 0; j != numProps; ++j)
        {
            float val;
            inFile.read(reinterpret_cast<char*>(&val), sizeof(float));
            propertiesData[j][i] = val;
        }
    }

    return vertexCount;
}

static glm::vec4 decodeVec4(uint32_t encoded)
{
    return glm::vec4(
        ((encoded >> 24u) & 0xFF) / 255.0f * 2.0f - 1.0f,
        ((encoded >> 16u) & 0xFF) / 255.0f * 2.0f - 1.0f,
        ((encoded >> 8u) & 0xFF) / 255.0f * 2.0f - 1.0f,
        (encoded & 0xFF) / 255.0f * 2.0f - 1.0f
    );
}

bool savePly(const char* filePath,
             const std::vector<float4>& position,
             const std::vector<float4>& scaleAndRotation,
             const std::vector<float4>& color)
{
    if (position.empty() || position.size() != scaleAndRotation.size() || position.size() != color.size())
    {
        return false;
    }

    std::ofstream outFile(filePath, std::ios_base::binary);
    if (!outFile.is_open())
    {
        return false;
    }

    const uint32_t vertexCount = static_cast<uint32_t>(position.size());
    const float SH_C0 = 0.28209479177387814f;

    outFile << "ply\n";
    outFile << "format binary_little_endian 1.0\n";
    outFile << "element vertex " << vertexCount << "\n";
    outFile << "property float x\n";
    outFile << "property float y\n";
    outFile << "property float z\n";
    outFile << "property float rot_0\n";
    outFile << "property float rot_1\n";
    outFile << "property float rot_2\n";
    outFile << "property float rot_3\n";
    outFile << "property float scale_0\n";
    outFile << "property float scale_1\n";
    outFile << "property float scale_2\n";
    outFile << "property float f_dc_0\n";
    outFile << "property float f_dc_1\n";
    outFile << "property float f_dc_2\n";
    outFile << "property float opacity\n";
    outFile << "end_header\n";

    for (uint32_t i = 0; i < vertexCount; ++i)
    {
        float x = position[i].x;
        float y = position[i].y;
        float z = position[i].z;
        outFile.write(reinterpret_cast<const char*>(&x), sizeof(float));
        outFile.write(reinterpret_cast<const char*>(&y), sizeof(float));
        outFile.write(reinterpret_cast<const char*>(&z), sizeof(float));

        uint32_t encodedRot = *reinterpret_cast<const uint32_t*>(&scaleAndRotation[i].w);
        glm::vec4 rot = decodeVec4(encodedRot);
        outFile.write(reinterpret_cast<const char*>(&rot.x), sizeof(float));
        outFile.write(reinterpret_cast<const char*>(&rot.y), sizeof(float));
        outFile.write(reinterpret_cast<const char*>(&rot.z), sizeof(float));
        outFile.write(reinterpret_cast<const char*>(&rot.w), sizeof(float));

        float sx = glm::log(scaleAndRotation[i].x);
        float sy = glm::log(scaleAndRotation[i].y);
        float sz = glm::log(scaleAndRotation[i].z);
        outFile.write(reinterpret_cast<const char*>(&sx), sizeof(float));
        outFile.write(reinterpret_cast<const char*>(&sy), sizeof(float));
        outFile.write(reinterpret_cast<const char*>(&sz), sizeof(float));

        float r = (color[i].x - 0.5f) / SH_C0;
        float g = (color[i].y - 0.5f) / SH_C0;
        float b = (color[i].z - 0.5f) / SH_C0;
        outFile.write(reinterpret_cast<const char*>(&r), sizeof(float));
        outFile.write(reinterpret_cast<const char*>(&g), sizeof(float));
        outFile.write(reinterpret_cast<const char*>(&b), sizeof(float));

        float opacity = position[i].w;
        opacity = glm::clamp(opacity, 0.001f, 0.999f);
        float logitOpacity = glm::log(opacity / (1.0f - opacity));
        outFile.write(reinterpret_cast<const char*>(&logitOpacity), sizeof(float));
    }

    outFile.close();
    return true;
}

int indexOf(const std::vector<std::string>& properties, const std::string& property)
{
    auto it = std::find(properties.begin(), properties.end(), property);
    if (it == properties.end())
    {
        throw std::invalid_argument("Missing property \"" + property + "\".");
    }
    return std::distance(properties.begin(), it);
}

uint32_t encodeVec4(const glm::vec4& v)
{
    auto clamped = glm::clamp(v, 0.0f, 1.0f);
    return (
        ((uint32_t) (clamped.x * 255.0f) << 24u) | 
        ((uint32_t) (clamped.y * 255.0f) << 16u) | 
        ((uint32_t) (clamped.z * 255.0f) << 8u ) | 
        ((uint32_t) (clamped.w * 255.0f)       ));
}

int parsePly(const char* filePath,
             std::vector<float4>& position,
             std::vector<float4>& scaleAndRotation,
             std::vector<float4>& color,
             std::vector<float>& sphericalHarmonics,
             int& sphericalHarmonicsDegree,
             int& sphericalHarmonicsCount,
             glm::vec3& boundsMin,
             glm::vec3& boundsMax)
{
    std::vector<std::string> properties;
    std::vector<std::vector<float>> propertiesData;
    auto vertexCount = parsePly(filePath, properties, propertiesData);

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

    position.resize(vertexCount);
    scaleAndRotation.resize(vertexCount);
    color.resize(vertexCount);

    auto extraSphericalHarmonicsCount{0};
    auto shOffsets = std::vector<int>();
    for (;;)
    {
        auto it =
            std::find(properties.begin(), properties.end(), "f_rest_" + std::to_string(extraSphericalHarmonicsCount));
        if (it == properties.end())
        {
            break;
        }
        else
        {
            shOffsets.push_back(std::distance(properties.begin(), it));
        }

        ++extraSphericalHarmonicsCount;
    }

    auto expectedSphericalHarmonicsCount{0};
    sphericalHarmonicsDegree = 0;
    while (expectedSphericalHarmonicsCount < extraSphericalHarmonicsCount)
    {
        expectedSphericalHarmonicsCount += (2 * (sphericalHarmonicsDegree + 1) + 1) * 3;
        ++sphericalHarmonicsDegree;
    }

    if (expectedSphericalHarmonicsCount != extraSphericalHarmonicsCount)
    {
        throw std::invalid_argument(
            "Expected degree " + std::to_string(sphericalHarmonicsDegree) + ", "
            + std::to_string(expectedSphericalHarmonicsCount) + " extra spherical harmonics." + " Found "
            + std::to_string(extraSphericalHarmonicsCount) + ".");
    }

    sphericalHarmonicsCount = extraSphericalHarmonicsCount + 3;

    if (sphericalHarmonicsDegree != 0)
    {
        sphericalHarmonics.resize(vertexCount * sphericalHarmonicsCount);

        for (auto i = 0; i != vertexCount; ++i)
        {
            auto ptr = reinterpret_cast<float*>(&(*sphericalHarmonics.begin()) + sphericalHarmonicsCount * i);

            for (auto j = 0; j != 3; ++j)
            {
                *ptr = propertiesData[offsets[10 + j]][i];
                ++ptr;
            }

            for (auto j = 0; j != extraSphericalHarmonicsCount; ++j)
            {
                *ptr = propertiesData[shOffsets[j]][i];
                ++ptr;
            }
        }

        auto rgbShCount = extraSphericalHarmonicsCount / 3;
        auto tmp = std::vector<float>(extraSphericalHarmonicsCount);

        for (auto i = 0; i != vertexCount; ++i)
        {
            auto start = sphericalHarmonicsCount * i + 3;
            for (auto j = 0; j != rgbShCount; ++j)
            {
                tmp[j * 3 + 0] = sphericalHarmonics[start + j];
                tmp[j * 3 + 1] = sphericalHarmonics[start + rgbShCount + j];
                tmp[j * 3 + 2] = sphericalHarmonics[start + rgbShCount * 2 + j];
            }
            for (auto j = 0; j != extraSphericalHarmonicsCount; ++j)
            {
                sphericalHarmonics[start + j] = tmp[j];
            }
        }
    }

    boundsMin = glm::vec3(std::numeric_limits<float>::max());
    boundsMax = glm::vec3(std::numeric_limits<float>::min());

    for (auto i = 0; i != vertexCount; ++i)
    {
        auto translation = glm::vec3(
            propertiesData[offsets[0]][i], 
            propertiesData[offsets[1]][i], 
            propertiesData[offsets[2]][i]);
        auto rotation = glm::quat(
            propertiesData[offsets[3]][i],
            propertiesData[offsets[4]][i],
            propertiesData[offsets[5]][i],
            propertiesData[offsets[6]][i]);
        auto scale = glm::vec3(
            propertiesData[offsets[7]][i], 
            propertiesData[offsets[8]][i], 
            propertiesData[offsets[9]][i]);
        auto sphericalHarmonicsVal = glm::vec3(
            propertiesData[offsets[10]][i], 
            propertiesData[offsets[11]][i], 
            propertiesData[offsets[12]][i]);
        auto opacity = propertiesData[offsets[13]][i];

        rotation = glm::normalize(rotation);
        scale = glm::exp(scale);
        opacity = sigmoid(opacity);

        boundsMin = glm::min(boundsMin, translation);
        boundsMax = glm::max(boundsMax, translation);

        const float SH_C0{0.28209479177387814f};
        auto rgb = sphericalHarmonicsVal * SH_C0 + 0.5f;

        auto quantizedRotation = encodeVec4((glm::vec4(rotation.x, rotation.y, rotation.z, rotation.w) + 1.0f) * 0.5f);

        position[i] = float4{translation.x, translation.y, translation.z, opacity};
        scaleAndRotation[i] = float4{scale.x, scale.y, scale.z, reinterpret_cast<float&>(quantizedRotation)};
        color[i] = float4{rgb.x, rgb.y, rgb.z, opacity};
    }

    return vertexCount;
}
