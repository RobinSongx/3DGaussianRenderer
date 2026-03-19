#include "glfw_window.h"

namespace gauss_render {

GlfwWindow::GlfwWindow(int width, int height, const std::string& title)
    : width_(width), height_(height) {
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    window_ = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window_) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window_);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        exit(EXIT_FAILURE);
    }

    glfwSetInputMode(window_, GLFW_STICKY_KEYS, GL_TRUE);
    glfwSetWindowUserPointer(window_, this);
}

GlfwWindow::~GlfwWindow() {
    glfwDestroyWindow(window_);
    glfwTerminate();
}

bool GlfwWindow::ShouldClose() const {
    return glfwWindowShouldClose(window_);
}

void GlfwWindow::SwapBuffers() {
    glfwSwapBuffers(window_);
}

void GlfwWindow::PollEvents() {
    glfwPollEvents();
}

void GlfwWindow::SetMouseMoveCallback(std::function<void(float, float)> callback) {
    mouse_move_callback_ = std::move(callback);
    glfwSetCursorPosCallback(window_, MouseMoveCallback);
}

void GlfwWindow::SetScrollCallback(std::function<void(float)> callback) {
    scroll_callback_ = std::move(callback);
    glfwSetScrollCallback(window_, ScrollCallback);
}

void GlfwWindow::MouseMoveCallback(GLFWwindow* window, double x, double y) {
    GlfwWindow* window_ptr = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(window));
    if (!window_ptr->mouse_move_callback_) {
        return;
    }

    if (window_ptr->first_mouse_) {
        window_ptr->last_x_ = x;
        window_ptr->last_y_ = y;
        window_ptr->first_mouse_ = false;
        return;
    }

    float dx = static_cast<float>(x - window_ptr->last_x_);
    float dy = static_cast<float>(window_ptr->last_y_ - y);
    window_ptr->last_x_ = x;
    window_ptr->last_y_ = y;

    window_ptr->mouse_move_callback_(dx, dy);
}

void GlfwWindow::ScrollCallback(GLFWwindow* window, double x, double y) {
    (void)x;
    GlfwWindow* window_ptr = static_cast<GlfwWindow*>(glfwGetWindowUserPointer(window));
    if (!window_ptr->scroll_callback_) {
        return;
    }
    window_ptr->scroll_callback_(static_cast<float>(y));
}

} // namespace gauss_render
