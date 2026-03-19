#ifndef GAUSS_RENDER_GAUSSIAN_MODEL_H_
#define GAUSS_RENDER_GAUSSIAN_MODEL_H_

#include "types.h"
#include <vector>
#include <glm/glm.hpp>

namespace gauss_render {

struct Gaussian {
    float position[3];
    float opacity;
    float scale[3];
    float rotation[4];
    float color[3];
};

class GaussianModel {
public:
    std::vector<Gaussian> gaussians;
    std::vector<float> sh_coefficients;
    int sh_degree = 0;
    int num_gaussians = 0;
    glm::vec3 bounds_min;
    glm::vec3 bounds_max;

    void PreprocessSh();
    const std::vector<float>& GetAlignedSh() const { return aligned_sh_; }
    glm::vec3 GetBoundsMin() const { return bounds_min; }
    glm::vec3 GetBoundsMax() const { return bounds_max; }

private:
    std::vector<float> aligned_sh_;
};

} // namespace gauss_render

#endif // GAUSS_RENDER_GAUSSIAN_MODEL_H_
