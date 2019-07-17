//! \file distribution_energy.h
//! Energy distributions that depend on incident particle energy

#ifndef OPENMC_DISTRIBUTION_ENERGY_H
#define OPENMC_DISTRIBUTION_ENERGY_H

#include <vector>

#include "xtensor/xtensor.hpp"
#include "hdf5.h"

#include "openmc/constants.h"
#include "openmc/endf.h"

namespace openmc {

//===============================================================================
//! Abstract class defining an energy distribution that is a function of the
//! incident energy of a projectile. Each derived type must implement a sample()
//! function that returns a sampled outgoing energy given an incoming energy
//===============================================================================

class EnergyDistribution {
public:
  virtual double sample(double E) const = 0;
  virtual ~EnergyDistribution() = default;
};

struct EnergyDistribution_ {
  enum class Type {
    none = 0, discrete_photon = 1, level = 2, continuous = 3, maxwell = 4, evaporation = 5, watt = 6
  };
};

//===============================================================================
//! Discrete photon energy distribution
//===============================================================================

class DiscretePhoton : public EnergyDistribution {
public:
  explicit DiscretePhoton(hid_t group);

  //! Sample energy distribution
  //! \param[in] E Incident particle energy in [eV]
  //! \return Sampled energy in [eV]
  double sample(double E) const;
// private:
  int primary_flag_; //!< Indicator of whether the photon is a primary or
                     //!< non-primary photon.
  double energy_; //!< Photon energy or binding energy
  double A_; //!< Atomic weight ratio of the target nuclide
};

struct DiscretePhoton_ {
  int primary_flag_; //!< Indicator of whether the photon is a primary or
                     //!< non-primary photon.
  double energy_; //!< Photon energy or binding energy
  double A_; //!< Atomic weight ratio of the target nuclide
};

//===============================================================================
//! Level inelastic scattering distribution
//===============================================================================

class LevelInelastic : public EnergyDistribution {
public:
  explicit LevelInelastic(hid_t group);

  //! Sample energy distribution
  //! \param[in] E Incident particle energy in [eV]
  //! \return Sampled energy in [eV]
  double sample(double E) const;
// private:
  double threshold_; //!< Energy threshold in lab, (A + 1)/A * |Q|
  double mass_ratio_; //!< (A/(A+1))^2
};

struct LevelInelastic_ {
  double threshold_; //!< Energy threshold in lab, (A + 1)/A * |Q|
  double mass_ratio_; //!< (A/(A+1))^2

  __device__ __forceinline__ LevelInelastic_() {}
};

//===============================================================================
//! An energy distribution represented as a tabular distribution with histogram
//! or linear-linear interpolation. This corresponds to ACE law 4, which NJOY
//! produces for a number of ENDF energy distributions.
//===============================================================================

class ContinuousTabular : public EnergyDistribution {
public:
  explicit ContinuousTabular(hid_t group);

  //! Sample energy distribution
  //! \param[in] E Incident particle energy in [eV]
  //! \return Sampled energy in [eV]
  double sample(double E) const;
// private:
  //! Outgoing energy for a single incoming energy
  struct CTTable {
    Interpolation interpolation; //!< Interpolation law
    int n_discrete; //!< Number of of discrete energies
    xt::xtensor<double, 1> e_out; //!< Outgoing energies in [eV]
    xt::xtensor<double, 1> p; //!< Probability density
    xt::xtensor<double, 1> c; //!< Cumulative distribution
  };

  int n_region_; //!< Number of inteprolation regions
  std::vector<int> breakpoints_; //!< Breakpoints between regions
  std::vector<Interpolation> interpolation_; //!< Interpolation laws
  std::vector<double> energy_; //!< Incident energy in [eV]
  std::vector<CTTable> distribution_; //!< Distributions for each incident energy
};

struct ContinuousTabular_ {

  struct CTTable_ {
    Interpolation interpolation; //!< Interpolation law
    int n_discrete; //!< Number of of discrete energies
    rtBufferId<double, 1> e_out; //!< Outgoing energies in [eV]
    rtBufferId<double, 1> p; //!< Probability density
    rtBufferId<double, 1> c; //!< Cumulative distribution

    __device__ __forceinline__ CTTable_() {}

    // __device__ __forceinline__ CTTable_(ContinuousTabular::CTTable &ct,
    //                                     rtBufferId<double, 1> e_out,
    //                                     rtBufferId<double, 1> p,
    //                                     rtBufferId<double, 1> c) {
    //   interpolation = ct.interpolation;
    //   n_discrete = ct.n_discrete;
    //   this->e_out = e_out;
    //   this->p = p;
    //   this->c = c;
    // }
  };

  int n_region_; //!< Number of inteprolation regions
  rtBufferId<int, 1> breakpoints_; //!< Breakpoints between regions
  rtBufferId<Interpolation, 1> interpolation_; //!< Interpolation laws
  rtBufferId<double, 1> energy_; //!< Incident energy in [eV]
  rtBufferId<CTTable_, 1> distribution_; //!< Distributions for each incident energy

  __device__ __forceinline__ ContinuousTabular_() {}

  // __device__ __forceinline__ ContinuousTabular_(ContinuousTabular &ct,
  //                                               rtBufferId<int, 1> breakpoints_,
  //                                               rtBufferId<Interpolation, 1> interpolation_,
  //                                               rtBufferId<double, 1> energy_,
  //                                               unsigned long energy_size,
  //                                               rtBufferId<CTTable_, 1> distribution_) {
  //   n_region_ = ct.n_region_;
  //   this->breakpoints_ = breakpoints_;
  //   this->interpolation_ = interpolation_;
  //   this->energy_ = energy_;
  //   this->energy_size = energy_size;
  //   this->distribution_ = distribution_;
  // }
};

//===============================================================================
//! Evaporation spectrum corresponding to ACE law 9 and ENDF File 5, LF=9.
//===============================================================================

class Evaporation : public EnergyDistribution {
public:
  explicit Evaporation(hid_t group);

  //! Sample energy distribution
  //! \param[in] E Incident particle energy in [eV]
  //! \return Sampled energy in [eV]
  double sample(double E) const;
private:
  Tabulated1D theta_; //!< Incoming energy dependent parameter
  double u_; //!< Restriction energy
};

//===============================================================================
//! Energy distribution of neutrons emitted from a Maxwell fission spectrum.
//! This corresponds to ACE law 7 and ENDF File 5, LF=7.
//===============================================================================

class MaxwellEnergy : public EnergyDistribution {
public:
  explicit MaxwellEnergy(hid_t group);

  //! Sample energy distribution
  //! \param[in] E Incident particle energy in [eV]
  //! \return Sampled energy in [eV]
  double sample(double E) const;
private:
  Tabulated1D theta_; //!< Incoming energy dependent parameter
  double u_; //!< Restriction energy
};

//===============================================================================
//! Energy distribution of neutrons emitted from a Watt fission spectrum. This
//! corresponds to ACE law 11 and ENDF File 5, LF=11.
//===============================================================================

class WattEnergy : public EnergyDistribution {
public:
  explicit WattEnergy(hid_t group);

  //! Sample energy distribution
  //! \param[in] E Incident particle energy in [eV]
  //! \return Sampled energy in [eV]
  double sample(double E) const;
private:
  Tabulated1D a_; //!< Energy-dependent 'a' parameter
  Tabulated1D b_; //!< Energy-dependent 'b' parameter
  double u_; //!< Restriction energy
};

} // namespace openmc

#endif // OPENMC_DISTRIBUTION_ENERGY_H
