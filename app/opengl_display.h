#ifndef GAUSS_RENDER_OPENGL_DISPLAY_H_
#define GAUSS_RENDER_OPENGL_DISPLAY_H_

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <vector>

namespace gauss_render {

class OpenglDisplay {
public:
    OpenglDisplay(int width, int height);
    ~OpenglDisplay();

    OpenglDisplay(const OpenglDisplay&) = delete;
    OpenglDisplay& operator=(const OpenglDisplay&) = delete;

    GLuint GetColorBufferId() const { return color_buffer_; }
    void Render();

private:
    GLuint CompileShader(const char* vertex_source, const char* fragment_source);
    void CreateQuad();

    int width_;
    int height_;
    GLuint shader_program_;
    GLuint vertex_array_;
    GLuint vertex_buffer_;
    GLuint uv_buffer_;
    GLuint index_buffer_;
    GLuint color_buffer_;
    GLint texture_location_;
};

} // namespace gauss_render

#endif // GAUSS_RENDER_OPENGL_DISPLAY_H_
