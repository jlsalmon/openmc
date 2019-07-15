#pragma once

#include <optix_world.h>

#include "endf.cu"
#include "secondary_uncorrelated.cu"
#include "secondary_kalbach.cu"
#include "random_lcg.cu"
#include "math_functions.cu"

#include "openmc/reaction_product.h"

__device__ __forceinline__
void _sample_angle_energy(const ReactionProduct_& rp, int i, double E_in, double& E_out, double& mu) {
  AngleEnergy_::Type type = rp.distribution_type;

  if (type == AngleEnergy_::Type::uncorrelated) {
    _sample_uncorrelated_angle_energy(rp.distribution_[i], E_in, E_out, mu);
  } else if (type == AngleEnergy_::Type::kalbach_mann) {
    // KalbachMann_& km = (KalbachMann_&) angle_energy;
    // _sample_kalbach_mann(rp.distribution_[i], E_in, E_out, mu);
    printf("TODO: sample kalbach-mann\n");
  }
}

__device__ __forceinline__
void _sample_reaction_product(const ReactionProduct_& rp, double E_in, double& E_out, double& mu)
{
  auto n = rp.applicability_size;
  if (n > 1) {
    double prob = 0.0; // FIXME
    double c = prn();
    for (int i = 0; i < n; ++i) {
      // Determine probability that i-th energy distribution is sampled
      // prob += rp.applicability_[i](E_in);
      prob += _tabulated_1d(rp.applicability_[i], E_in);

      // If i-th distribution is sampled, sample energy from the distribution
      if (c <= prob) {
        _sample_angle_energy(rp, i, E_in, E_out, mu);
        // rp.distribution_[i].sample(E_in, E_out, mu);
        // _sample_uncorrelated_angle_energy(rp.distribution_[i], E_in, E_out, mu);
        break;
      }
    }
  } else {
    // If only one distribution is present, go ahead and sample it
    // rp.distribution_[0].sample(E_in, E_out, mu);
    _sample_angle_energy(rp, 0, E_in, E_out, mu);
    // _sample_uncorrelated_angle_energy(rp.distribution_[0], E_in, E_out, mu);
  }
}