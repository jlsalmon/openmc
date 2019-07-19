#pragma once

#include <optix_world.h>

#include "endf.cu"
#include "secondary_uncorrelated.cu"
#include "secondary_kalbach.cu"
#include "random_lcg.cu"
#include "math_functions.cu"

#include "openmc/reaction_product.h"

__device__ __forceinline__
void _sample_angle_energy(const ReactionProduct_& rp, int i, float E_in, float& E_out, float& mu) {
  AngleEnergy_::Type type = rp.distribution_type;

  if (type == AngleEnergy_::Type::uncorrelated) {
    // rtPrintf("UAE.distribution buffer id: %d\n", rp.distribution_.getId());
    _sample_uncorrelated_angle_energy(rp.distribution_uae[i], E_in, E_out, mu);
  } else if (type == AngleEnergy_::Type::kalbach_mann) {
    _sample_kalbach_mann(rp.distribution_km[i], E_in, E_out, mu);
  } else {
    printf("ERROR: unknown AngleEnergy type\n");
  }
}

__device__ __forceinline__
void _sample_reaction_product(const ReactionProduct_& rp, float E_in, float& E_out, float& mu)
{
  auto n = rp.applicability_.size();
  if (n > 1) {
    float prob = 0.0f; // FIXME
    float c = prn();
    for (int i = 0; i < n; ++i) {
      // Determine probability that i-th energy distribution is sampled
      // prob += rp.applicability_[i](E_in);
      rtPrintf("rp.applicability_ buffer id: %d\n", rp.applicability_.getId());
      prob += _tabulated_1d(rp.applicability_[i], E_in);

      // If i-th distribution is sampled, sample energy from the distribution
      if (c <= prob) {
        rtPrintf("E_out before sampling angle energy (1): %f\n", E_out);
        _sample_angle_energy(rp, i, E_in, E_out, mu);
        rtPrintf("E_out after sampling angle energy (1): %f\n", E_out);
        // rp.distribution_[i].sample(E_in, E_out, mu);
        // _sample_uncorrelated_angle_energy(rp.distribution_[i], E_in, E_out, mu);
        break;
      }
    }
  } else {
    // If only one distribution is present, go ahead and sample it
    // rp.distribution_[0].sample(E_in, E_out, mu);
    rtPrintf("E_out before sampling angle energy (2): %f\n", E_out);
    _sample_angle_energy(rp, 0, E_in, E_out, mu);
    rtPrintf("E_out after sampling angle energy (2): %f\n", E_out);
    // _sample_uncorrelated_angle_energy(rp.distribution_[0], E_in, E_out, mu);
  }
}