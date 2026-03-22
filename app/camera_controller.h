#ifndef GAUSS_RENDER_CAMERA_CONTROLLER_H_
#define GAUSS_RENDER_CAMERA_CONTROLLER_H_

#include "../core/camera.h"
#include <glm/glm.hpp>

namespace gauss_render {

// 相机控制器
// 处理用户输入（鼠标移动、滚轮），更新相机位姿
// 围绕场景中心点旋转，支持缩放
class CameraController {
public:
    // 构造：绑定到相机
    explicit CameraController(Camera* camera);

    // 更新相机（每一帧调用）
    void Update(float delta_time);
    // 处理鼠标移动：旋转相机
    void OnMouseMove(float dx, float dy);
    // 处理滚轮：缩放相机距离
    void OnScroll(float dy);
    // 设置场景包围盒，用于限制相机距离
    void SetBounds(const glm::vec3& min_bounds, const glm::vec3& max_bounds);

private:
    // 根据旋转角度和距离更新相机位姿
    void UpdateCameraPose();

    Camera* camera_;           // 绑定的相机
    float yaw_ = 0.0f;         // 偏航角（绕Y轴旋转）
    float pitch_ = 0.0f;        // 俯仰角（绕X轴旋转）
    float distance_ = 2.0f;     // 相机到中心的距离
    glm::vec3 center_;         // 旋转中心（场景中心）
    glm::vec3 bounds_min_;      // 场景包围盒最小点
    glm::vec3 bounds_max_;      // 场景包围盒最大点
    float sensitivity_ = 0.01f; // 鼠标旋转灵敏度
    float scroll_speed_ = 0.1f; // 滚轮缩放灵敏度
};

} // namespace gauss_render

#endif // GAUSS_RENDER_CAMERA_CONTROLLER_H_
