#pragma once

#include <optix_world.h>

#include "random_lcg.cu"
#include "math_functions.cu"

#include "openmc/distribution_multi.h"

using namespace openmc;

__device__ __forceinline__
Direction_ _sample_isotropic(Isotropic_& angle) {
  float phi = 2.0f*M_PIf*prn();
  float mu = 2.0f*prn() - 1.0f;
  return {mu, sqrtf(1.0f - mu*mu) * cosf(phi),
          sqrtf(1.0f - mu*mu) * sinf(phi)};
}