#ifndef GAUSS_RENDER_GLFW_WINDOW_H_
#define GAUSS_RENDER_GLFW_WINDOW_H_

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <functional>
#include <string>

namespace gauss_render {

// GLFW窗口封装
// 创建窗口，处理事件，回调鼠标移动和滚轮
class GlfwWindow {
public:
    // 构造：创建指定大小和标题的窗口
    GlfwWindow(int width, int height, const std::string& title);
    // 析构：销毁窗口，终止GLFW
    ~GlfwWindow();

    // 禁止拷贝
    GlfwWindow(const GlfwWindow&) = delete;
    GlfwWindow& operator=(const GlfwWindow&) = delete;

    // 检查窗口是否应该关闭
    bool ShouldClose() const;
    // 交换前后缓冲区，显示新帧
    void SwapBuffers();
    // 处理事件
    void PollEvents();

    // 获取窗口宽度
    int GetWidth() const { return width_; }
    // 获取窗口高度
    int GetHeight() const { return height_; }
    // 获取GLFW窗口句柄
    GLFWwindow* GetHandle() const { return window_; }

    // 设置鼠标移动回调
    void SetMouseMoveCallback(std::function<void(float, float)> callback);
    // 设置滚轮回调
    void SetScrollCallback(std::function<void(float)> callback);

private:
    // 静态回调函数：GLFW需要静态函数
    static void MouseMoveCallback(GLFWwindow* window, double x, double y);
    static void ScrollCallback(GLFWwindow* window, double x, double y);

    int width_;               // 窗口宽度
    int height_;              // 窗口高度
    GLFWwindow* window_ = nullptr;  // GLFW窗口句柄
    double last_x_ = 0.0;     // 上一帧鼠标X坐标
    double last_y_ = 0.0;     // 上一帧鼠标Y坐标
    bool first_mouse_ = true;  // 是否是第一次鼠标移动

    std::function<void(float, float)> mouse_move_callback_;  // 鼠标移动回调
    std::function<void(float)> scroll_callback_;             // 滚轮回调
};

} // namespace gauss_render

#endif // GAUSS_RENDER_GLFW_WINDOW_H_
