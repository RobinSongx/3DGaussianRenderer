#ifndef GAUSS_RENDER_CAMERA_H_
#define GAUSS_RENDER_CAMERA_H_

#include "types.h"
#include <glm/glm.hpp>

namespace gauss_render {

class Camera {
public:
    Camera() = default;

    void SetFovY(float fov_y_rad);
    void SetResolution(int width, int height);
    void SetNearFar(float near, float far);
    void SetPose(const glm::vec3& position, const glm::vec3& look_at);

    glm::mat4 GetViewProjection() const { return view_projection_; }
    glm::mat4 GetProjection() const { return projection_; }
    glm::mat4 GetView() const { return view_; }
    glm::vec3 GetPosition() const { return position_; }
    float GetAspect() const { return aspect_; }
    glm::vec2 GetFovCotangent() const { return fov_cotangent_; }
    glm::vec2 GetDepthScaleBias() const { return depth_scale_bias_; }
    int GetWidth() const { return width_; }
    int GetHeight() const { return height_; }

private:
    void UpdateMatrices();

    glm::vec3 position_;
    glm::mat4 view_;
    glm::mat4 projection_;
    glm::mat4 view_projection_;
    glm::vec2 fov_cotangent_;
    glm::vec2 depth_scale_bias_;
    float aspect_ = 1.0f;
    float fov_y_ = glm::radians(50.0f);
    float near_ = 0.01f;
    float far_ = 100.0f;
    int width_ = kDefaultScreenWidth;
    int height_ = kDefaultScreenHeight;
};

} // namespace gauss_render

#endif // GAUSS_RENDER_CAMERA_H_
