#ifndef GAUSS_RENDER_CAMERA_CONTROLLER_H_
#define GAUSS_RENDER_CAMERA_CONTROLLER_H_

#include "../core/camera.h"
#include <glm/glm.hpp>

namespace gauss_render {

class CameraController {
public:
    explicit CameraController(Camera* camera);

    void Update(float delta_time);
    void OnMouseMove(float dx, float dy);
    void OnScroll(float dy);
    void SetBounds(const glm::vec3& min_bounds, const glm::vec3& max_bounds);

private:
    void UpdateCameraPose();

    Camera* camera_;
    float yaw_ = 0.0f;
    float pitch_ = 0.0f;
    float distance_ = 2.0f;
    glm::vec3 center_;
    glm::vec3 bounds_min_;
    glm::vec3 bounds_max_;
    float sensitivity_ = 0.01f;
    float scroll_speed_ = 0.1f;
};

} // namespace gauss_render

#endif // GAUSS_RENDER_CAMERA_CONTROLLER_H_
