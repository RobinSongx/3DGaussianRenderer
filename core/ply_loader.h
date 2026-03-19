#ifndef GAUSS_RENDER_PLY_LOADER_H_
#define GAUSS_RENDER_PLY_LOADER_H_

#include "gaussian_model.h"
#include <string>

namespace gauss_render {

int LoadPly(const std::string& path, GaussianModel& model);

} // namespace gauss_render

#endif // GAUSS_RENDER_PLY_LOADER_H_
