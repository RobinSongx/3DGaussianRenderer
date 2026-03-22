#include "camera_controller.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

namespace gauss_render {

// 构造函数：绑定到相机，初始化默认参数
CameraController::CameraController(Camera* camera)
    : camera_(camera) {
    center_ = glm::vec3(0.0f);
    bounds_min_ = glm::vec3(-5.0f);
    bounds_max_ = glm::vec3(5.0f);
}

// 处理鼠标移动
// dx, dy: 鼠标移动增量
// 更新旋转角度，限制俯仰角在[-89, 89]度避免万向节死锁
void CameraController::OnMouseMove(float dx, float dy) {
    yaw_ += dx * sensitivity_;
    pitch_ += dy * sensitivity_;

    // 限制俯仰角，避免翻转
    if (pitch_ > glm::radians(89.0f)) {
        pitch_ = glm::radians(89.0f);
    }
    if (pitch_ < glm::radians(-89.0f)) {
        pitch_ = glm::radians(-89.0f);
    }

    UpdateCameraPose();
}

// 处理滚轮滚动
// dy: 滚轮增量，正表示向前滚（放大）
// 更新相机到中心的距离，限制在合理范围内
void CameraController::OnScroll(float dy) {
    distance_ -= dy * scroll_speed_ * distance_;
    distance_ = glm::clamp(distance_, distance_ * 0.1f, distance_ * 10.0f);
    distance_ = glm::clamp(distance_, 0.1f, 100.0f);
    UpdateCameraPose();
}

// 设置场景包围盒
// 根据包围盒计算中心位置和初始相机距离
void CameraController::SetBounds(const glm::vec3& min_bounds, const glm::vec3& max_bounds) {
    bounds_min_ = min_bounds;
    bounds_max_ = max_bounds;
    center_ = (min_bounds + max_bounds) * 0.5f;

    // 根据场景大小设置初始相机距离
    float size = glm::length(max_bounds - min_bounds);
    distance_ = size * 0.8f;
    UpdateCameraPose();
}

// 更新相机（每帧调用）
// 当前没有连续移动，所以不需要做任何事
void CameraController::Update(float delta_time) {
    (void)delta_time;
}

// 根据当前旋转角度和距离更新相机位姿
// 计算相机位置：从中心看向负方向，距离distance_
void CameraController::UpdateCameraPose() {
    // 计算相机看向的方向（球坐标转笛卡尔坐标）
    glm::vec3 direction;
    direction.x = cos(yaw_) * cos(pitch_);
    direction.y = sin(pitch_);
    direction.z = sin(yaw_) * cos(pitch_);
    direction = glm::normalize(direction);

    // 相机位置 = 中心 - 方向 * 距离
    glm::vec3 position = center_ - direction * distance_;
    camera_->SetPose(position, center_);
}

} // namespace gauss_render
