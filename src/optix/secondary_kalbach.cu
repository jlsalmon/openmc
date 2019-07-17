#pragma once

#include <optix_world.h>

#include "random_lcg.cu"
#include "math_functions.cu"

#include "openmc/secondary_kalbach.h"


__device__ __forceinline__
void _sample_kalbach_mann(const KalbachMann_& km, double E_in, double& E_out, double& mu) {
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< REMOVE THIS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  // Before the secondary distribution refactor, an isotropic polar cosine was
  // always sampled but then overwritten with the polar cosine sampled from the
  // correlated distribution. To preserve the random number stream, we keep
  // this dummy sampling here but can remove it later (will change answers)
  mu = 2.0*prn() - 1.0;
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< REMOVE THIS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

  // Find energy bin and calculate interpolation factor -- if the energy is
  // outside the range of the tabulated energies, choose the first or last bins
  auto n_energy_in = km.energy_.size();
  int i;
  double r;
  if (E_in < km.energy_[0]) {
    i = 0;
    r = 0.0;
  } else if (E_in > km.energy_[n_energy_in - 1]) {
    i = n_energy_in - 2;
    r = 1.0;
  } else {
    // i = lower_bound_index(energy_.begin(), energy_.end(), E_in);
    i = _lower_bound(0, km.energy_.size(), km.energy_, E_in);
    r = (E_in - km.energy_[i]) / (km.energy_[i+1] - km.energy_[i]);
  }

  // Sample between the ith and [i+1]th bin
  int l = r > prn() ? i + 1 : i;

  // Interpolation for energy E1 and EK
  int n_energy_out = km.distribution_[i].e_out.size();
  int n_discrete = km.distribution_[i].n_discrete;
  double E_i_1 = km.distribution_[i].e_out[n_discrete];
  double E_i_K = km.distribution_[i].e_out[n_energy_out - 1];

  n_energy_out = km.distribution_[i+1].e_out.size();
  n_discrete = km.distribution_[i+1].n_discrete;
  double E_i1_1 = km.distribution_[i+1].e_out[n_discrete];
  double E_i1_K = km.distribution_[i+1].e_out[n_energy_out - 1];

  double E_1 = E_i_1 + r*(E_i1_1 - E_i_1);
  double E_K = E_i_K + r*(E_i1_K - E_i_K);

  // Determine outgoing energy bin
  n_energy_out = km.distribution_[l].e_out.size();
  n_discrete = km.distribution_[l].n_discrete;
  double r1 = prn();
  double c_k = km.distribution_[l].c[0];
  int k = 0;
  int end = n_energy_out - 2;

  // Discrete portion
  for (int j = 0; j < n_discrete; ++j) {
    k = j;
    c_k = km.distribution_[l].c[k];
    if (r1 < c_k) {
      end = j;
      break;
    }
  }

  // Continuous portion
  double c_k1;
  for (int j = n_discrete; j < end; ++j) {
    k = j;
    c_k1 = km.distribution_[l].c[k+1];
    if (r1 < c_k1) break;
    k = j + 1;
    c_k = c_k1;
  }

  double E_l_k = km.distribution_[l].e_out[k];
  double p_l_k = km.distribution_[l].p[k];
  double km_r, km_a;
  if (km.distribution_[l].interpolation == Interpolation::histogram) {
    // Histogram interpolation
    if (p_l_k > 0.0 && k >= n_discrete) {
      E_out = E_l_k + (r1 - c_k)/p_l_k;
    } else {
      E_out = E_l_k;
    }

    // Determine Kalbach-Mann parameters
    km_r = km.distribution_[l].r[k];
    km_a = km.distribution_[l].a[k];

  } else {
    // Linear-linear interpolation
    double E_l_k1 = km.distribution_[l].e_out[k+1];
    double p_l_k1 = km.distribution_[l].p[k+1];

    double frac = (p_l_k1 - p_l_k)/(E_l_k1 - E_l_k);
    if (frac == 0.0) {
      E_out = E_l_k + (r1 - c_k)/p_l_k;
    } else {
      E_out = E_l_k + (sqrtf(fmaxf(0.0, p_l_k*p_l_k +
                                               2.0*frac*(r1 - c_k))) - p_l_k)/frac;
    }

    // Determine Kalbach-Mann parameters
    km_r = km.distribution_[l].r[k] + (E_out - E_l_k)/(E_l_k1 - E_l_k) *
                                   (km.distribution_[l].r[k+1] - km.distribution_[l].r[k]);
    km_a = km.distribution_[l].a[k] + (E_out - E_l_k)/(E_l_k1 - E_l_k) *
                                   (km.distribution_[l].a[k+1] - km.distribution_[l].a[k]);
  }

  // Now interpolate between incident energy bins i and i + 1
  if (k >= n_discrete) {
    if (l == i) {
      E_out = E_1 + (E_out - E_i_1)*(E_K - E_1)/(E_i_K - E_i_1);
    } else {
      E_out = E_1 + (E_out - E_i1_1)*(E_K - E_1)/(E_i1_K - E_i1_1);
    }
  }

  // Sampled correlated angle from Kalbach-Mann parameters
  if (prn() > km_r) {
    double T = (2.0*prn() - 1.0) * sinhf(km_a);
    mu = logf(T + sqrt(T*T + 1.0))/km_a;
  } else {
    double r1 = prn();
    mu = logf(r1*expf(km_a) + (1.0 - r1)*expf(-km_a))/km_a;
  }
}
