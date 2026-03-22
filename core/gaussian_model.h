#ifndef GAUSS_RENDER_GAUSSIAN_MODEL_H_
#define GAUSS_RENDER_GAUSSIAN_MODEL_H_

#include "types.h"
#include <vector>
#include <glm/glm.hpp>

namespace gauss_render {

// 单个高斯的参数（CPU端存储）
struct Gaussian {
    float position[3];    // 世界空间位置 (x, y, z)
    float opacity;         // 不透明度（激活前）
    float scale[3];        // 三个轴的缩放因子
    float rotation[4];     // 旋转四元数 (w, x, y, z)
    float color[3];        // 静态颜色（SH阶数为0时使用）
};

// 整个高斯场景模型
// 存储所有高斯参数，负责预处理（球谐系数重排）
class GaussianModel {
public:
    std::vector<Gaussian> gaussians;       // 所有高斯参数列表
    std::vector<float> sh_coefficients;    // 原始球谐系数（degrees > 0时有效）
    int sh_degree = 0;                     // 球谐最大阶数
    int num_gaussians = 0;                 // 高斯总数
    glm::vec3 bounds_min;                 // 场景包围盒最小点
    glm::vec3 bounds_max;                 // 场景包围盒最大点

    // 预处理球谐系数：重排列以适应GPU合并访问
    void PreprocessSh();
    // 获取预处理对齐后的球谐系数
    const std::vector<float>& GetAlignedSh() const { return aligned_sh_; }
    // 获取场景包围盒最小点
    glm::vec3 GetBoundsMin() const { return bounds_min; }
    // 获取场景包围盒最大点
    glm::vec3 GetBoundsMax() const { return bounds_max; }

private:
    std::vector<float> aligned_sh_;  // 对齐后的球谐系数（用于GPU）
};

} // namespace gauss_render

#endif // GAUSS_RENDER_GAUSSIAN_MODEL_H_
