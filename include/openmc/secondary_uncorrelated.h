//! \file secondary_uncorrelated.h
//! Uncorrelated angle-energy distribution

#ifndef OPENMC_SECONDARY_UNCORRELATED_H
#define OPENMC_SECONDARY_UNCORRELATED_H

#include <memory>
#include <vector>
#include <optix_world.h>

#include "hdf5.h"

#include "openmc/angle_energy.h"
#include "openmc/distribution_angle.h"
#include "openmc/distribution_energy.h"

namespace openmc {

//==============================================================================
//! Uncorrelated angle-energy distribution. This corresponds to when an energy
//! distribution is given in ENDF File 5/6 and an angular distribution is given
//! in ENDF File 4.
//==============================================================================

class UncorrelatedAngleEnergy : public AngleEnergy {
public:
  explicit UncorrelatedAngleEnergy(hid_t group);

  //! Sample distribution for an angle and energy
  //! \param[in] E_in Incoming energy in [eV]
  //! \param[out] E_out Outgoing energy in [eV]
  //! \param[out] mu Outgoing cosine with respect to current direction
  void sample(double E_in, double& E_out, double& mu) const;

  // Accessors
  AngleDistribution& angle() { return angle_; }
  bool& fission() { return fission_; }
// private:
  AngleDistribution angle_; //!< Angle distribution
  std::unique_ptr<EnergyDistribution> energy_; //!< Energy distribution
  bool fission_ {false}; //!< Whether distribution is use for fission
};

struct UncorrelatedAngleEnergy_ {
  AngleDistribution_ angle_; //!< Angle distribution
  bool angle_empty;
  ContinuousTabular_ energy_ct; //!< Energy distribution
  LevelInelastic_ energy_li;
  DiscretePhoton_ energy_dp;
  EnergyDistribution_::Type energy_type;
  bool fission_; //!< Whether distribution is use for fission

  bool& fission() { return fission_; }

  __device__ __forceinline__ UncorrelatedAngleEnergy_() {};

  // __device__ __forceinline__ UncorrelatedAngleEnergy_(UncorrelatedAngleEnergy *uae,
  //                                                     AngleDistribution_ angle_,
  //                                                     ContinuousTabular_ energy_) {
  //   this->angle_ = angle_;
  //   this->angle_empty = uae->angle_.empty();
  //   this->energy_ = energy_;
  //   this->fission_ = uae->fission_;
  //   this->type = Type::uncorrelated;
  // }
};

} // namespace openmc

#endif // OPENMC_SECONDARY_UNCORRELATED_H
