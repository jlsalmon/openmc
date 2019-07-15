#pragma once

#include <optix_world.h>

#include "random_lcg.cu"
#include "math_functions.cu"

#include "openmc/distribution_energy.h"

using namespace openmc;


__device__ __forceinline__
double _sample_continuous_tabular_distribution(const ContinuousTabular_& ct, double E)
{
  // Read number of interpolation regions and incoming energies
  bool histogram_interp;
  if (ct.n_region_ == 1) {
    histogram_interp = (ct.interpolation_[0] == Interpolation::histogram);
  } else {
    histogram_interp = false;
  }

  // Find energy bin and calculate interpolation factor -- if the energy is
  // outside the range of the tabulated energies, choose the first or last bins
  auto n_energy_in = ct.energy_size;
  int i;
  double r;
  if (E < ct.energy_[0]) {
    i = 0;
    r = 0.0;
  } else if (E > ct.energy_[n_energy_in - 1]) {
    i = n_energy_in - 2;
    r = 1.0;
  } else {
    // i = lower_bound_index(energy_.begin(), energy_.end(), E);
    i = _lower_bound(0, ct.energy_size, ct.energy_, E);
    r = (E - ct.energy_[i]) / (ct.energy_[i+1] - ct.energy_[i]);
  }

  // Sample between the ith and [i+1]th bin
  int l;
  if (histogram_interp) {
    l = i;
  } else {
    l = r > prn() ? i + 1 : i;
  }

  // Interpolation for energy E1 and EK
  int n_energy_out = ct.distribution_[i].e_out.size();
  int n_discrete = ct.distribution_[i].n_discrete;
  double E_i_1 = ct.distribution_[i].e_out[n_discrete];
  double E_i_K = ct.distribution_[i].e_out[n_energy_out - 1];

  n_energy_out = ct.distribution_[i+1].e_out.size();
  n_discrete = ct.distribution_[i+1].n_discrete;
  double E_i1_1 = ct.distribution_[i+1].e_out[n_discrete];
  double E_i1_K = ct.distribution_[i+1].e_out[n_energy_out - 1];

  double E_1 = E_i_1 + r*(E_i1_1 - E_i_1);
  double E_K = E_i_K + r*(E_i1_K - E_i_K);

  // Determine outgoing energy bin
  n_energy_out = ct.distribution_[l].e_out.size();
  n_discrete = ct.distribution_[l].n_discrete;
  double r1 = prn();
  double c_k = ct.distribution_[l].c[0];
  int k = 0;
  int end = n_energy_out - 2;

  // Discrete portion
  for (int j = 0; j < n_discrete; ++j) {
    k = j;
    c_k = ct.distribution_[l].c[k];
    if (r1 < c_k) {
      end = j;
      break;
    }
  }

  // Continuous portion
  double c_k1;
  for (int j = n_discrete; j < end; ++j) {
    k = j;
    c_k1 = ct.distribution_[l].c[k+1];
    if (r1 < c_k1) break;
    k = j + 1;
    c_k = c_k1;
  }

  double E_l_k = ct.distribution_[l].e_out[k];
  double p_l_k = ct.distribution_[l].p[k];
  double E_out;
  if (ct.distribution_[l].interpolation == Interpolation::histogram) {
    // Histogram interpolation
    if (p_l_k > 0.0 && k >= n_discrete) {
      E_out = E_l_k + (r1 - c_k)/p_l_k;
    } else {
      E_out = E_l_k;
    }

  } else if (ct.distribution_[l].interpolation == Interpolation::lin_lin) {
    // Linear-linear interpolation
    double E_l_k1 = ct.distribution_[l].e_out[k+1];
    double p_l_k1 = ct.distribution_[l].p[k+1];

    double frac = (p_l_k1 - p_l_k)/(E_l_k1 - E_l_k);
    if (frac == 0.0) {
      E_out = E_l_k + (r1 - c_k)/p_l_k;
    } else {
      E_out = E_l_k + (sqrt(fmax(0.0, p_l_k*p_l_k +
                                               2.0*frac*(r1 - c_k))) - p_l_k)/frac;
    }
  } else {
    // throw std::runtime_error{"Unexpected interpolation for continuous energy "
    //                          "distribution."};
    printf("ERROR: Unexpected interpolation for continuous energy distribution.\n");
  }

  // Now interpolate between incident energy bins i and i + 1
  if (!histogram_interp && n_energy_out > 1 && k >= n_discrete) {
    if (l == i) {
      return E_1 + (E_out - E_i_1)*(E_K - E_1)/(E_i_K - E_i_1);
    } else {
      return E_1 + (E_out - E_i1_1)*(E_K - E_1)/(E_i1_K - E_i1_1);
    }
  } else {
    return E_out;
  }
}

__device__ __forceinline__
double _sample_discrete_photon_distribution(const DiscretePhoton_& dp, double E) {
  // TODO
  printf("DISCRETE PHOTON\n");
}

__device__ __forceinline__
double _sample_level_inelastic_distribution(const LevelInelastic_& dp, double E) {
  // TODO
  printf("LEVEL INELASTIC\n");
}