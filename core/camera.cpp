#include "camera.h"
#include <glm/gtc/matrix_transform.hpp>

namespace gauss_render {

// 设置垂直方向视场角
void Camera::SetFovY(float fov_y_rad) {
    fov_y_ = fov_y_rad;
    UpdateMatrices();
}

// 设置输出图像分辨率
void Camera::SetResolution(int width, int height) {
    width_ = width;
    height_ = height;
    aspect_ = static_cast<float>(width) / static_cast<float>(height);
    UpdateMatrices();
}

// 设置近裁剪面和远裁剪面
void Camera::SetNearFar(float near, float far) {
    near_ = near;
    far_ = far;
    UpdateMatrices();
}

// 设置相机位姿
// position: 相机世界空间位置
// look_at: 相机看向的目标点
void Camera::SetPose(const glm::vec3& position, const glm::vec3& look_at) {
    position_ = position;
    glm::vec3 direction = glm::normalize(look_at - position);
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    view_ = glm::lookAt(position, look_at, up);
    UpdateMatrices();
}

// 更新所有矩阵和预计算参数
// 每次修改相机参数后需要调用
void Camera::UpdateMatrices() {
    // 计算投影矩阵
    projection_ = glm::perspective(fov_y_, aspect_, near_, far_);
    // 计算视图投影矩阵
    view_projection_ = projection_ * view_;

    // 预计算视场角余切，用于投影计算
    float cotangent_y = 1.0f / glm::tan(fov_y_ * 0.5f);
    float cotangent_x = cotangent_y / aspect_;
    fov_cotangent_ = glm::vec2(cotangent_x, cotangent_y);

    // 预计算深度缩放和偏移，将深度从[near, far]映射到[-1, 1]
    float scale_z = -2.0f / (far_ - near_);
    float translation_z = -(far_ + near_) / (far_ - near_);
    depth_scale_bias_ = glm::vec2(scale_z, translation_z);
}

} // namespace gauss_render
