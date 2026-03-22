#ifndef GAUSS_RENDER_OPENGL_DISPLAY_H_
#define GAUSS_RENDER_OPENGL_DISPLAY_H_

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <vector>

namespace gauss_render {

// OpenGL显示
// 使用一个全屏四边形(quad)绘制CUDA渲染的图像
// 颜色缓冲区用OpenGL PBO + 纹理，和CUDA互操作
class OpenglDisplay {
public:
    // 构造：创建着色器，顶点缓冲区，颜色PBO
    OpenglDisplay(int width, int height);
    // 析构：销毁所有OpenGL对象
    ~OpenglDisplay();

    // 禁止拷贝
    OpenglDisplay(const OpenglDisplay&) = delete;
    OpenglDisplay& operator=(const OpenglDisplay&) = delete;

    // 获取颜色缓冲区（PBO）的ID，供CUDA写入
    GLuint GetColorBufferId() const { return color_buffer_; }
    // 绘制全屏四边形，显示CUDA渲染的图像
    void Render();

private:
    // 编译着色器
    GLuint CompileShader(const char* vertex_source, const char* fragment_source);
    // 创建全屏四边形的顶点数据
    void CreateQuad();

    int width_;                // 显示宽度
    int height_;               // 显示高度
    GLuint shader_program_;    // 着色器程序
    GLuint vertex_array_;      // VAO
    GLuint vertex_buffer_;     // 顶点坐标缓冲区
    GLuint uv_buffer_;         // UV坐标缓冲区
    GLuint index_buffer_;      // 索引缓冲区
    GLuint color_buffer_;      // 颜色缓冲区（PBO，供CUDA写入）
    GLint texture_location_;   // 纹理采样器位置
};

} // namespace gauss_render

#endif // GAUSS_RENDER_OPENGL_DISPLAY_H_
