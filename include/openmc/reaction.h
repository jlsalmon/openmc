//! \file reaction.h
//! Data for an incident neutron reaction

#ifndef OPENMC_REACTION_H
#define OPENMC_REACTION_H

#include <string>
#include <vector>

#include "hdf5.h"

#include "openmc/reaction_product.h"

namespace openmc {

//==============================================================================
//! Data for a single reaction including cross sections (possibly at multiple
//! temperatures) and reaction products (with secondary angle-energy
//! distributions)
//==============================================================================

class Reaction {
public:
  //! Construct reaction from HDF5 data
  //! \param[in] group HDF5 group containing reaction data
  //! \param[in] temperatures Desired temperatures for cross sections
  explicit Reaction(hid_t group, const std::vector<int> &temperatures);

  //! Cross section at a single temperature
  struct TemperatureXS {
    int threshold;
    std::vector<double> value;
  };

  int mt_;             //!< ENDF MT value
  double q_value_;     //!< Reaction Q value in [eV]
  bool scatter_in_cm_; //!< scattering system in center-of-mass?
  bool redundant_;     //!< redundant reaction?
  std::vector<TemperatureXS> xs_; //!< Cross section at each temperature
  std::vector<ReactionProduct> products_; //!< Reaction products
};

struct Reaction_ {

  struct TemperatureXS_ {
    int threshold;
    rtBufferId<double, 1> value_;

    __device__ __forceinline__ TemperatureXS_() {}

    __device__ __forceinline__ TemperatureXS_(Reaction::TemperatureXS *t,
                                              rtBufferId<double, 1> value_) {
      threshold = t->threshold;
      this->value_ = value_;
    }
  };

  int mt_;             //!< ENDF MT value
  double q_value_;     //!< Reaction Q value in [eV]
  bool scatter_in_cm_; //!< scattering system in center-of-mass?
  bool redundant_;     //!< redundant reaction?
  rtBufferId<TemperatureXS_, 1> xs_; //!< Cross section at each temperature
  rtBufferId<ReactionProduct_, 1> products_; //!< Reaction products

  __device__ __forceinline__ Reaction_() {}

  __device__ __forceinline__ Reaction_(Reaction *r,
                                       rtBufferId<TemperatureXS_, 1> xs_,
                                       rtBufferId<ReactionProduct_, 1> products_) {
    mt_ = r->mt_;
    q_value_ = r->q_value_;
    scatter_in_cm_ = r->scatter_in_cm_;
    redundant_ = r->redundant_;
    this->xs_ = xs_;
    this->products_ = products_;
  }
};

//==============================================================================
// Non-member functions
//==============================================================================

std::string reaction_name(int mt);

} // namespace openmc

#endif // OPENMC_REACTION_H
