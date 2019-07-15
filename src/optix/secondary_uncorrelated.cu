#pragma once

#include <optix_world.h>

#include "distribution_angle.cu"
#include "distribution_energy.cu"
#include "random_lcg.cu"

#include "openmc/secondary_uncorrelated.h"


__device__ __forceinline__
double _sample_energy_distribution(const UncorrelatedAngleEnergy_& uae, double E_in) {
  // EnergyDistribution_::Type type = energy.type;
  double E_out;

  if (uae.energy_type == EnergyDistribution_::Type::continuous) {
    E_out = _sample_continuous_tabular_distribution(uae.energy_, E_in);
  } else if (uae.energy_type == EnergyDistribution_::Type::discrete_photon) {
    printf("TODO: sample discrete photon\n");
    // E_out = _sample_discrete_photon_distribution(dp, E_in);
  } else if (uae.energy_type == EnergyDistribution_::Type::level) {
    // E_out = _sample_level_inelastic_distribution(li, E_in);
    printf("TODO: sample level inelastic\n");
    E_out = E_in;
  }

  return E_out;
}

__device__ __forceinline__
void _sample_uncorrelated_angle_energy(const UncorrelatedAngleEnergy_& uae, double E_in, double& E_out, double& mu)
{
  // Sample cosine of scattering angle
  if (uae.fission_) {
    // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< REMOVE THIS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // For fission, the angle is not used, so just assign a dummy value
    mu = 1.0;
    // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< REMOVE THIS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  } else if (!uae.angle_empty) {
    // mu = angle_.sample(E_in);
    mu = _sample_angle_distribution(uae.angle_, E_in);
  } else {
    // no angle distribution given => assume isotropic for all energies
    mu = 2.0*prn() - 1.0;
  }

  // // Sample outgoing energy
  // E_out = energy_->sample(E_in);
  // E_out = _sample_continuous_tabular_distribution(uae.energy_, E_in);
  E_out = _sample_energy_distribution(uae, E_in);
}