#include "opengl_display.h"
#include <cstdio>
#include <vector>

#define STRINGIFY(A) #A

namespace gauss_render {

namespace {

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

const char* kFragmentShaderText = STRINGIFY(
    #version 330 core\n
    in vec2 uv;\n
    out vec4 color;\n
    uniform sampler2D textureSampler;\n
    void main() {\n
        color = texture(textureSampler, uv).rgba;\n    }\n
);

constexpr uint32_t kQuadIndices[] = {0, 1, 2, 2, 3, 0};
constexpr glm::vec3 kQuadVertices[] = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};
constexpr glm::vec2 kQuadUvs[] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};

} // namespace

OpenglDisplay::OpenglDisplay(int width, int height)
    : width_(width), height_(height) {
    shader_program_ = CompileShader(kVertexShaderText, kFragmentShaderText);
    texture_location_ = glGetUniformLocation(shader_program_, "textureSampler");

    CreateQuad();

    glGenBuffers(1, &color_buffer_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, color_buffer_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

OpenglDisplay::~OpenglDisplay() {
    glDeleteBuffers(1, &color_buffer_);
    glDeleteBuffers(1, &index_buffer_);
    glDeleteBuffers(1, &uv_buffer_);
    glDeleteBuffers(1, &vertex_buffer_);
    glDeleteVertexArrays(1, &vertex_array_);
    glDeleteProgram(shader_program_);
}

GLuint OpenglDisplay::CompileShader(const char* vertex_source, const char* fragment_source) {
    GLuint vertex_id = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragment_id = glCreateShader(GL_FRAGMENT_SHADER);
    GLint result = GL_FALSE;
    int info_length;

    glShaderSource(vertex_id, 1, &vertex_source, nullptr);
    glCompileShader(vertex_id);
    glGetShaderiv(vertex_id, GL_COMPILE_STATUS, &result);
    glGetShaderiv(vertex_id, GL_INFO_LOG_LENGTH, &info_length);
    if (info_length > 0) {
        std::vector<char> error_msg(info_length + 1);
        glGetShaderInfoLog(vertex_id, info_length, nullptr, error_msg.data());
        fprintf(stderr, "%s\n", error_msg.data());
    }

    glShaderSource(fragment_id, 1, &fragment_source, nullptr);
    glCompileShader(fragment_id);
    glGetShaderiv(fragment_id, GL_COMPILE_STATUS, &result);
    glGetShaderiv(fragment_id, GL_INFO_LOG_LENGTH, &info_length);
    if (info_length > 0) {
        std::vector<char> error_msg(info_length + 1);
        glGetShaderInfoLog(fragment_id, info_length, nullptr, error_msg.data());
        fprintf(stderr, "%s\n", error_msg.data());
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_id);
    glAttachShader(program, fragment_id);
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &result);
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &info_length);
    if (info_length > 0) {
        std::vector<char> error_msg(info_length + 1);
        glGetProgramInfoLog(program, info_length, nullptr, error_msg.data());
        fprintf(stderr, "%s\n", error_msg.data());
    }

    glDetachShader(program, vertex_id);
    glDetachShader(program, fragment_id);
    glDeleteShader(vertex_id);
    glDeleteShader(fragment_id);

    return program;
}

void OpenglDisplay::CreateQuad() {
    glGenVertexArrays(1, &vertex_array_);
    glBindVertexArray(vertex_array_);

    glGenBuffers(1, &vertex_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(kQuadVertices), kQuadVertices, GL_STATIC_DRAW);

    glGenBuffers(1, &uv_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, uv_buffer_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(kQuadUvs), kQuadUvs, GL_STATIC_DRAW);

    glGenBuffers(1, &index_buffer_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(kQuadIndices), kQuadIndices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindVertexArray(0);
}

void OpenglDisplay::Render() {
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);

    glUseProgram(shader_program_);
    glEnable(GL_TEXTURE_2D);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, color_buffer_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    glUniform1i(texture_location_, 0);

    glBindVertexArray(vertex_array_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

    glDisable(GL_TEXTURE_2D);
}

} // namespace gauss_render
