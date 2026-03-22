#include "glfw_window.h"

namespace gauss_render {

// 构造函数：创建GLFW窗口
// width, height: 窗口大小
// title: 窗口标题
GlfwWindow::GlfwWindow(int width, int height, const std::string& title)
    : width_(width), height_(height) {
    // 初始化GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        exit(EXIT_FAILURE);
    }

    // 设置OpenGL版本 3.3 core profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // 创建窗口
    window_ = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window_) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // 将OpenGL上下文绑定到当前线程
    glfwMakeContextCurrent(window_);

    // 初始化GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        exit(EXIT_FAILURE);
    }

    // 启用粘滞按键，让glfwGetKey一直返回按下状态
    glfwSetInputMode(window_, GLFW_STICKY_KEYS, GL_TRUE);
    // 将this指针保存到GLFW窗口用户数据中，供静态回调使用
    glfwSetWindowUserPointer(window_, this);
}

// 析构函数：销毁窗口，终止GLFW
GlfwWindow::~GlfwWindow() {
    glfwDestroyWindow(window_);
    glfwTerminate();
}

// 检查窗口是否应该关闭（用户点击关闭按钮）
bool GlfwWindow::ShouldClose() const {
    return glfwWindowShouldClose(window_);
}

// 交换前后缓冲区，显示新渲染的帧
void GlfwWindow::SwapBuffers() {
    glfwSwapBuffers(window_);
}

// 处理窗口事件（鼠标、键盘、关闭等）
void GlfwWindow::PollEvents() {
    glfwPollEvents();
}

// 设置鼠标移动回调
void GlfwWindow::SetMouseMoveCallback(std::function<void(float, float)> callback) {
    mouse_move_callback_ = std::move(callback);
    glfwSetCursorPosCallback(window_, MouseMoveCallback);
}

// 设置滚轮回调
void GlfwWindow::SetScrollCallback(std::function<void(float)> callback) {
    scroll_callback_ = std::move(callback);
    glfwSetScrollCallback(window_, ScrollCallback);
}

// 静态鼠标移动回调
// 从GLFW窗口获取this指针，然后调用用户回调
// 计算鼠标增量dx, dy
void GlfwWindow::MouseMoveCallback(GLFWwindow* window, double x, double y) {
    GlfwWindow* window_ptr = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(window));
    if (!window_ptr->mouse_move_callback_) {
        return;
    }

    // 第一次鼠标移动只记录初始位置，不产生回调
    if (window_ptr->first_mouse_) {
        window_ptr->last_x_ = x;
        window_ptr->last_y_ = y;
        window_ptr->first_mouse_ = false;
        return;
    }

    // 计算鼠标移动增量（dy是y方向减，因为GLFW y向下为正）
    float dx = static_cast<float>(x - window_ptr->last_x_);
    float dy = static_cast<float>(window_ptr->last_y_ - y);
    window_ptr->last_x_ = x;
    window_ptr->last_y_ = y;

    window_ptr->mouse_move_callback_(dx, dy);
}

// 静态滚轮回调
void GlfwWindow::ScrollCallback(GLFWwindow* window, double x, double y) {
    (void)x;
    GlfwWindow* window_ptr = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(window));
    if (!window_ptr->scroll_callback_) {
        return;
    }
    // y是滚动方向，向上滚为正，对应放大（减小相机距离）
    window_ptr->scroll_callback_(static_cast<float>(y));
}

} // namespace gauss_render
