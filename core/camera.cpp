#include "camera.h"
#include <glm/gtc/matrix_transform.hpp>

namespace gauss_render {

void Camera::SetFovY(float fov_y_rad) {
    fov_y_ = fov_y_rad;
    UpdateMatrices();
}

void Camera::SetResolution(int width, int height) {
    width_ = width;
    height_ = height;
    aspect_ = static_cast<float>(width) / static_cast<float>(height);
    UpdateMatrices();
}

void Camera::SetNearFar(float near, float far) {
    near_ = near;
    far_ = far;
    UpdateMatrices();
}

void Camera::SetPose(const glm::vec3& position, const glm::vec3& look_at) {
    position_ = position;
    glm::vec3 direction = glm::normalize(look_at - position);
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    view_ = glm::lookAt(position, look_at, up);
    UpdateMatrices();
}

void Camera::UpdateMatrices() {
    projection_ = glm::perspective(fov_y_, aspect_, near_, far_);
    view_projection_ = projection_ * view_;

    float cotangent_y = 1.0f / glm::tan(fov_y_ * 0.5f);
    float cotangent_x = cotangent_y / aspect_;
    fov_cotangent_ = glm::vec2(cotangent_x, cotangent_y);

    float scale_z = -2.0f / (far_ - near_);
    float translation_z = -(far_ + near_) / (far_ - near_);
    depth_scale_bias_ = glm::vec2(scale_z, translation_z);
}

} // namespace gauss_render
