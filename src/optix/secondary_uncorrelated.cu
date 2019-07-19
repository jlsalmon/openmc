#pragma once

#include <optix_world.h>

#include "distribution_angle.cu"
#include "distribution_energy.cu"
#include "random_lcg.cu"

#include "openmc/secondary_uncorrelated.h"


__device__ __forceinline__
float _sample_energy_distribution(const UncorrelatedAngleEnergy_& uae, float E_in) {
  float E_out;

  if (uae.energy_type == EnergyDistribution_::Type::continuous) {
    E_out = _sample_continuous_tabular_distribution(uae.energy_ct, E_in);
  } else if (uae.energy_type == EnergyDistribution_::Type::discrete_photon) {
    E_out = _sample_discrete_photon_distribution(uae.energy_dp, E_in);
  } else if (uae.energy_type == EnergyDistribution_::Type::level) {
    E_out = _sample_level_inelastic_distribution(uae.energy_li, E_in);
  } else {
    // printf("ERROR: Unsupported energy distribution type %d\n", uae.energy_type);
    E_out = E_in;
  }

  return E_out;
}

__device__ __forceinline__
void _sample_uncorrelated_angle_energy(const UncorrelatedAngleEnergy_& uae, float E_in, float& E_out, float& mu)
{
  // Sample cosine of scattering angle
  if (uae.fission_) {
    // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< REMOVE THIS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // For fission, the angle is not used, so just assign a dummy value
    mu = 1.0f;
    // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< REMOVE THIS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  } else if (!uae.angle_empty) {
    // mu = angle_.sample(E_in);
    mu = _sample_angle_distribution(uae.angle_, E_in);
  } else {
    // no angle distribution given => assume isotropic for all energies
    mu = 2.0f*prn() - 1.0f;
  }

  // // Sample outgoing energy
  // E_out = energy_->sample(E_in);
  // E_out = _sample_continuous_tabular_distribution(uae.energy_, E_in);
  E_out = _sample_energy_distribution(uae, E_in);
}