#pragma once

#include <optix_world.h>

#include "distribution.cu"
#include "math_functions.cu"

#include "openmc/distribution_angle.h"

using namespace openmc;

__device__ __forceinline__
double _sample_angle_distribution(const AngleDistribution_& ad, double E)
{
  // Determine number of incoming energies
  auto n = ad.energy_size;

  // Find energy bin and calculate interpolation factor -- if the energy is
  // outside the range of the tabulated energies, choose the first or last bins
  int i;
  double r;
  if (E < ad.energy_[0]) {
    i = 0;
    r = 0.0;
  } else if (E > ad.energy_[n - 1]) {
    i = n - 2;
    r = 1.0;
  } else {
    // i = lower_bound_index(energy_.begin(), energy_.end(), E);
    i = _lower_bound(0, ad.energy_size, ad.energy_, E);
    r = (E - ad.energy_[i])/(ad.energy_[i+1] - ad.energy_[i]);
  }

  // Sample between the ith and (i+1)th bin
  if (r > prn()) ++i;

  // Sample i-th distribution
  // double mu = distribution_[i]->sample();
  double mu = _sample_tabular(ad.distribution_[i]);

  // Make sure mu is in range [-1,1] and return
  if (std::abs(mu) > 1.0) mu = std::copysign(1.0, mu);
  return mu;
}