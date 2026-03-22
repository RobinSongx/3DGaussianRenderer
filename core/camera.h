#ifndef GAUSS_RENDER_CAMERA_H_
#define GAUSS_RENDER_CAMERA_H_

#include "types.h"
#include <glm/glm.hpp>

namespace gauss_render {

// 相机参数类
// 存储相机的所有参数（位置、朝向、投影矩阵等），纯数据结构，不包含交互逻辑
class Camera {
public:
    Camera() = default;

    // 设置垂直方向视场角（弧度制）
    void SetFovY(float fov_y_rad);
    // 设置输出图像分辨率
    void SetResolution(int width, int height);
    // 设置近裁剪面和远裁剪面
    void SetNearFar(float near, float far);
    // 设置相机位姿（位置和看向的点）
    void SetPose(const glm::vec3& position, const glm::vec3& look_at);

    // 获取视图投影矩阵
    glm::mat4 GetViewProjection() const { return view_projection_; }
    // 获取投影矩阵
    glm::mat4 GetProjection() const { return projection_; }
    // 获取视图矩阵
    glm::mat4 GetView() const { return view_; }
    // 获取相机世界空间位置
    glm::vec3 GetPosition() const { return position_; }
    // 获取图像宽高比
    float GetAspect() const { return aspect_; }
    // 获取视场角余切（用于投影计算）
    glm::vec2 GetFovCotangent() const { return fov_cotangent_; }
    // 获取深度缩放和偏移（用于投影深度到[0,1]）
    glm::vec2 GetDepthScaleBias() const { return depth_scale_bias_; }
    // 获取图像宽度
    int GetWidth() const { return width_; }
    // 获取图像高度
    int GetHeight() const { return height_; }

private:
    // 更新投影和视图投影矩阵
    void UpdateMatrices();

    glm::vec3 position_;          // 相机世界空间位置
    glm::mat4 view_;              // 视图矩阵（世界空间 -> 相机空间）
    glm::mat4 projection_;        // 投影矩阵（相机空间 -> 裁剪空间）
    glm::mat4 view_projection_;   // 视图投影矩阵（世界空间 -> 裁剪空间）
    glm::vec2 fov_cotangent_;     // 视场角余切 (cotan(fov_x), cotan(fov_y))
    glm::vec2 depth_scale_bias_;  // 深度缩放和偏移 (scale, bias)
    float aspect_ = 1.0f;         // 图像宽高比
    float fov_y_ = glm::radians(50.0f);  // 垂直方向视场角（默认50度）
    float near_ = 0.01f;          // 近裁剪面
    float far_ = 100.0f;          // 远裁剪面
    int width_ = kDefaultScreenWidth;    // 图像宽度（默认1024）
    int height_ = kDefaultScreenHeight;  // 图像高度（默认1024）
};

} // namespace gauss_render

#endif // GAUSS_RENDER_CAMERA_H_
