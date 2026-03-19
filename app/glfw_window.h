#ifndef GAUSS_RENDER_GLFW_WINDOW_H_
#define GAUSS_RENDER_GLFW_WINDOW_H_

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <functional>
#include <string>

namespace gauss_render {

class GlfwWindow {
public:
    GlfwWindow(int width, int height, const std::string& title);
    ~GlfwWindow();

    GlfwWindow(const GlfwWindow&) = delete;
    GlfwWindow& operator=(const GlfwWindow&) = delete;

    bool ShouldClose() const;
    void SwapBuffers();
    void PollEvents();

    int GetWidth() const { return width_; }
    int GetHeight() const { return height_; }
    GLFWwindow* GetHandle() const { return window_; }

    void SetMouseMoveCallback(std::function<void(float, float)> callback);
    void SetScrollCallback(std::function<void(float)> callback);

private:
    static void MouseMoveCallback(GLFWwindow* window, double x, double y);
    static void ScrollCallback(GLFWwindow* window, double x, double y);

    int width_;
    int height_;
    GLFWwindow* window_ = nullptr;
    double last_x_ = 0.0;
    double last_y_ = 0.0;
    bool first_mouse_ = true;

    std::function<void(float, float)> mouse_move_callback_;
    std::function<void(float)> scroll_callback_;
};

} // namespace gauss_render

#endif // GAUSS_RENDER_GLFW_WINDOW_H_
