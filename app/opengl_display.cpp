#include "opengl_display.h"
#include <cstdio>
#include <vector>

#define STRINGIFY(A) #A

namespace gauss_render {

namespace {

// 顶点着色器：直接传递位置和UV坐标
const char* kVertexShaderText = STRINGIFY(
    #version 330 core\n
    layout(location = 0) in vec3 vertexPosition_modelspace;\n
    layout(location = 1) in vec2 vertexUv;\n
    out vec2 uv;\n
    void main() {\n
        gl_Position = vec4(vertexPosition_modelspace, 1);\n
        uv = vertexUv;\n
    }\n
);

// 片段着色器：从纹理采样颜色直接输出
const char* kFragmentShaderText = STRINGIFY(
    #version 330 core\n
    in vec2 uv;\n
    out vec4 color;\n
    uniform sampler2D textureSampler;\n
    void main() {\n
        color = texture(textureSampler, uv).rgba;\n    }\n
);

// 全屏四边形顶点坐标（归一化设备坐标）
constexpr glm::vec3 kQuadVertices[] = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};
// 纹理UV坐标
constexpr glm::vec2 kQuadUvs[] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
// 三角形索引（两个三角形组成四边形）
constexpr uint32_t kQuadIndices[] = {0, 1, 2, 2, 3, 0};

} // namespace

// 构造函数：创建OpenGL显示
// width, height: 显示分辨率
OpenglDisplay::OpenglDisplay(int width, int height)
    : width_(width), height_(height) {
    // 编译顶点和片段着色器
    shader_program_ = CompileShader(kVertexShaderText, kFragmentShaderText);
    // 获取纹理采样器 uniform 位置
    texture_location_ = glGetUniformLocation(shader_program_, "textureSampler");

    // 创建全屏四边形顶点数据
    CreateQuad();

    // 创建颜色缓冲区（像素缓冲区对象PBO）
    // CUDA直接写入这个缓冲区，然后纹理从这里读取
    glGenBuffers(1, &color_buffer_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, color_buffer_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

// 析构函数：销毁所有OpenGL对象
OpenglDisplay::~OpenglDisplay() {
    glDeleteBuffers(1, &color_buffer_);
    glDeleteBuffers(1, &index_buffer_);
    glDeleteBuffers(1, &uv_buffer_);
    glDeleteBuffers(1, &vertex_buffer_);
    glDeleteVertexArrays(1, &vertex_array_);
    glDeleteProgram(shader_program_);
}

// 编译着色器
// vertex_source: 顶点着色器源码
// fragment_source: 片段着色器源码
// 返回: 编译好的着色器程序ID
GLuint OpenglDisplay::CompileShader(const char* vertex_source, const char* fragment_source) {
    GLuint vertex_id = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragment_id = glCreateShader(GL_FRAGMENT_SHADER);
    GLint result = GL_FALSE;
    int info_length;

    // 编译顶点着色器
    glShaderSource(vertex_id, 1, &vertex_source, nullptr);
    glCompileShader(vertex_id);
    glGetShaderiv(vertex_id, GL_COMPILE_STATUS, &result);
    glGetShaderiv(vertex_id, GL_INFO_LOG_LENGTH, &info_length);
    if (info_length > 0) {
        std::vector<char> error_msg(info_length + 1);
        glGetShaderInfoLog(vertex_id, info_length, nullptr, error_msg.data());
        fprintf(stderr, "%s\n", error_msg.data());
    }

    // 编译片段着色器
    glShaderSource(fragment_id, 1, &fragment_source, nullptr);
    glCompileShader(fragment_id);
    glGetShaderiv(fragment_id, GL_COMPILE_STATUS, &result);
    glGetShaderiv(fragment_id, GL_INFO_LOG_LENGTH, &info_length);
    if (info_length > 0) {
        std::vector<char> error_msg(info_length + 1);
        glGetShaderInfoLog(fragment_id, info_length, nullptr, error_msg.data());
        fprintf(stderr, "%s\n", error_msg.data());
    }

    // 链接着色器程序
    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_id);
    glAttachShader(program, fragment_id);
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &result);
    if (!result) {
        fprintf(stderr, "Shader link failed!\n");
    }
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &info_length);
    if (info_length > 0) {
        std::vector<char> error_msg(info_length + 1);
        glGetProgramInfoLog(program, info_length, nullptr, error_msg.data());
        fprintf(stderr, "Link log: %s\n", error_msg.data());
    }

    // 清理
    glDetachShader(program, vertex_id);
    glDetachShader(program, fragment_id);
    glDeleteShader(vertex_id);
    glDeleteShader(fragment_id);

    return program;
}

// 创建全屏四边形顶点数组
// 设置VAO、VBO、EBO
void OpenglDisplay::CreateQuad() {
    glGenVertexArrays(1, &vertex_array_);
    glBindVertexArray(vertex_array_);

    // 顶点坐标
    glGenBuffers(1, &vertex_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(kQuadVertices), kQuadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    // UV坐标
    glGenBuffers(1, &uv_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, uv_buffer_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(kQuadUvs), kQuadUvs, GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    // 索引
    glGenBuffers(1, &index_buffer_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(kQuadIndices), kQuadIndices, GL_STATIC_DRAW);

    glBindVertexArray(0);
}

// 绘制全屏四边形，显示CUDA渲染的图像
void OpenglDisplay::Render() {
    // 清屏
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);

    // 使用着色器程序
    glUseProgram(shader_program_);

    // 绑定纹理
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // 从PBO创建纹理，数据在PBO中，CUDA已经写入
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, color_buffer_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    // 设置纹理单位
    glUniform1i(texture_location_, 0);

    // 绘制四边形
    glBindVertexArray(vertex_array_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
}

} // namespace gauss_render
