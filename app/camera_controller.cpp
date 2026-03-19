#include "camera_controller.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

namespace gauss_render {

CameraController::CameraController(Camera* camera)
    : camera_(camera) {
    center_ = glm::vec3(0.0f);
    bounds_min_ = glm::vec3(-5.0f);
    bounds_max_ = glm::vec3(5.0f);
}

void CameraController::OnMouseMove(float dx, float dy) {
    yaw_ += dx * sensitivity_;
    pitch_ += dy * sensitivity_;

    if (pitch_ > glm::radians(89.0f)) {
        pitch_ = glm::radians(89.0f);
    }
    if (pitch_ < glm::radians(-89.0f)) {
        pitch_ = glm::radians(-89.0f);
    }

    UpdateCameraPose();
}

void CameraController::OnScroll(float dy) {
    distance_ -= dy * scroll_speed_ * distance_;
    distance_ = glm::clamp(distance_, distance_ * 0.1f, distance_ * 10.0f);
    distance_ = glm::clamp(distance_, 0.1f, 100.0f);
    UpdateCameraPose();
}

void CameraController::SetBounds(const glm::vec3& min_bounds, const glm::vec3& max_bounds) {
    bounds_min_ = min_bounds;
    bounds_max_ = max_bounds;
    center_ = (min_bounds + max_bounds) * 0.5f;

    float size = glm::length(max_bounds - min_bounds);
    distance_ = size * 0.8f;
    UpdateCameraPose();
}

void CameraController::Update(float delta_time) {
    (void)delta_time;
}

void CameraController::UpdateCameraPose() {
    glm::vec3 direction;
    direction.x = cos(yaw_) * cos(pitch_);
    direction.y = sin(pitch_);
    direction.z = sin(yaw_) * cos(pitch_);
    direction = glm::normalize(direction);

    glm::vec3 position = center_ - direction * distance_;
    camera_->SetPose(position, center_);
}

} // namespace gauss_render
