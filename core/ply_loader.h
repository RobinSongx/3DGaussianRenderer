#ifndef GAUSS_RENDER_PLY_LOADER_H_
#define GAUSS_RENDER_PLY_LOADER_H_

#include "gaussian_model.h"
#include <string>

namespace gauss_render {

// 从PLY文件加载高斯模型
// path: PLY文件路径
// model: 输出模型数据
// 返回值: 高斯数量
int LoadPly(const std::string& path, GaussianModel& model);

} // namespace gauss_render

#endif // GAUSS_RENDER_PLY_LOADER_H_
