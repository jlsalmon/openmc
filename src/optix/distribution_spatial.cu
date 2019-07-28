#pragma once

#include <optix_world.h>

#include "random_lcg.cu"
#include "math_functions.cu"

#include "openmc/distribution_spatial.h"

using namespace openmc;

__device__ __forceinline__
Position_ _sample_spatial_box(SpatialBox_& space) {
  Position_ xi {prn(), prn(), prn()};
  return space.lower_left_ + xi*(space.upper_right_ - space.lower_left_);
}