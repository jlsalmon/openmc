#pragma once

#include <optix_world.h>

#include "math_functions.cu"

#include "openmc/endf.h"

__device__ __forceinline__
float _polynomial(const Polynomial_& p, float x)
{
  // Use Horner's rule to evaluate polynomial. Note that coefficients are
  // ordered in increasing powers of x.
  float y = 0.0f;
  for (int i = 0; i < p.num_coeffs; ++i) {
    auto c = p.coef_[i];
    y = y*x + c;
  }
  return y;
}

__device__ __forceinline__
float _tabulated_1d(const Tabulated1D_& t, float x)
{
  // rtPrintf("Tabulated1D_.x_ buffer id: %d\n", t.x_.getId());
  // rtPrintf("Tabulated1D_.int_ buffer id: %d\n", t.int_.getId());
  // rtPrintf("Tabulated1D_.nbt_ buffer id: %d\n", t.nbt_.getId());
  // rtPrintf("Tabulated1D_.y_ buffer id: %d\n", t.y_.getId());

  // find which bin the abscissa is in -- if the abscissa is outside the
  // tabulated range, the first or last point is chosen, i.e. no interpolation
  // is done outside the energy range
  int i;
  if (x < t.x_[0]) {
    return t.y_[0];
  } else if (x > t.x_[t.n_pairs_ - 1]) {
    return t.y_[t.n_pairs_ - 1];
  } else {
    // i = lower_bound_index(x_.begin(), x_.end(), x);
    i = _lower_bound(0, t.n_pairs_, t.x_, x);
  }

  // determine interpolation scheme
  Interpolation interp;
  if (t.n_regions_ == 0) {
    interp = Interpolation::lin_lin;
  } else {
    interp = t.int_[0];
    for (int j = 0; j < t.n_regions_; ++j) {
      if (i < t.nbt_[j]) {
        interp = t.int_[j];
        break;
      }
    }
  }

  // handle special case of histogram interpolation
  if (interp == Interpolation::histogram) return t.y_[i];

  // determine bounding values
  float x0 = t.x_[i];
  float x1 = t.x_[i + 1];
  float y0 = t.y_[i];
  float y1 = t.y_[i + 1];

  // determine interpolation factor and interpolated value
  float r;
  switch (interp) {
    case Interpolation::lin_lin:
      r = (x - x0)/(x1 - x0);
      return y0 + r*(y1 - y0);
    case Interpolation::lin_log:
      r = log(x/x0)/log(x1/x0);
      return y0 + r*(y1 - y0);
    case Interpolation::log_lin:
      r = (x - x0)/(x1 - x0);
      return y0*exp(r*log(y1/y0));
    case Interpolation::log_log:
      r = log(x/x0)/log(x1/x0);
      return y0*exp(r*log(y1/y0));
    default:
      printf("ERROR: Invalid interpolation scheme.");
      // throw std::runtime_error{"Invalid interpolation scheme."};
  }
}