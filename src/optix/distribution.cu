#pragma once

#include <optix_world.h>

#include "random_lcg.cu"

#include "openmc/distribution.h"

using namespace openmc;

__device__ __forceinline__
double _sample_tabular(const Tabular_& t)
{
  rtPrintf("Tabular_.x_ buffer id: %d\n", t.x_.getId());
  rtPrintf("Tabular_.c_ buffer id: %d\n", t.c_.getId());
  rtPrintf("Tabular_.p_ buffer id: %d\n", t.p_.getId());

  // Sample value of CDF
  double c = prn();

  // Find first CDF bin which is above the sampled value
  double c_i = t.c_[0];
  int i;
  size_t n = t.c_.size();
  for (i = 0; i < n - 1; ++i) {
    if (c <= t.c_[i+1]) break;
    c_i = t.c_[i+1];
  }

  // Determine bounding PDF values
  double x_i = t.x_[i];
  double p_i = t.p_[i];

  if (t.interp_ == Interpolation::histogram) {
    // Histogram interpolation
    if (p_i > 0.0) {
      return x_i + (c - c_i)/p_i;
    } else {
      return x_i;
    }
  } else {
    // Linear-linear interpolation
    double x_i1 = t.x_[i + 1];
    double p_i1 = t.p_[i + 1];

    double m = (p_i1 - p_i)/(x_i1 - x_i);
    if (m == 0.0) {
      return x_i + (c - c_i)/p_i;
    } else {
      return x_i + (sqrt(fmax(0.0, p_i*p_i + 2*m*(c - c_i))) - p_i)/m;
    }
  }
}