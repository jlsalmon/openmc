#pragma once

#include <optix_world.h>

#include "random_lcg.cu"
#include "math_functions.cu"

#include "openmc/distribution.h"

using namespace openmc;

__device__ __forceinline__
float _sample_tabular(const Tabular_& distribution)
{
  // rtPrintf("Tabular_.x_ buffer id: %d\n", t.x_.getId());
  // rtPrintf("Tabular_.c_ buffer id: %d\n", t.c_.getId());
  // rtPrintf("Tabular_.p_ buffer id: %d\n", t.p_.getId());

  // Sample value of CDF
  float c = prn();

  // Find first CDF bin which is above the sampled value
  float c_i = distribution.c_[0];
  int i;
  size_t n = distribution.c_.size();
  for (i = 0; i < n - 1; ++i) {
    if (c <= distribution.c_[i+1]) break;
    c_i = distribution.c_[i+1];
  }

  // Determine bounding PDF values
  float x_i = distribution.x_[i];
  float p_i = distribution.p_[i];

  if (distribution.interp_ == Interpolation::histogram) {
    // Histogram interpolation
    if (p_i > 0.0f) {
      return x_i + (c - c_i)/p_i;
    } else {
      return x_i;
    }
  } else {
    // Linear-linear interpolation
    float x_i1 = distribution.x_[i + 1];
    float p_i1 = distribution.p_[i + 1];

    float m = (p_i1 - p_i)/(x_i1 - x_i);
    if (m == 0.0f) {
      return x_i + (c - c_i)/p_i;
    } else {
      return x_i + (sqrtf(fmaxf(0.0f, p_i*p_i + 2*m*(c - c_i))) - p_i)/m;
    }
  }
}

__device__ __forceinline__
float _sample_watt(const Watt_& distribution) {
  return watt_spectrum(distribution.a_, distribution.b_);
}