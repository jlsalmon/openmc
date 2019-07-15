//! \file endf.h
//! Classes and functions related to the ENDF-6 format

#ifndef OPENMC_ENDF_H
#define OPENMC_ENDF_H

#include <memory>
#include <vector>

#include <optix_world.h>

#include "hdf5.h"

#include "openmc/constants.h"

namespace openmc {

//! Convert integer representing interpolation law to enum
//! \param[in] i Intereger (e.g. 1=histogram, 2=lin-lin)
//! \return Corresponding enum value
Interpolation int2interp(int i);

//! Determine whether MT number corresponds to a fission reaction
//! \param[in] MT ENDF MT value
//! \return Whether corresponding reaction is a fission reaction
bool is_fission(int MT);

//! Determine if a given MT number is that of a disappearance reaction, i.e., a
//! reaction with no neutron in the exit channel
//! \param[in] MT ENDF MT value
//! \return Whether corresponding reaction is a disappearance reaction
bool is_disappearance(int MT);

//! Determine if a given MT number is that of an inelastic scattering reaction
//! \param[in] MT ENDF MT value
//! \return Whether corresponding reaction is an inelastic scattering reaction
bool is_inelastic_scatter(int MT);

//==============================================================================
//! Abstract one-dimensional function
//==============================================================================

class Function1D {
public:
  virtual double operator()(double x) const = 0;
  virtual ~Function1D() = default;
};

//==============================================================================
//! One-dimensional function expressed as a polynomial
//==============================================================================

class Polynomial : public Function1D {
public:
  //! Construct polynomial from HDF5 data
  //! \param[in] dset Dataset containing coefficients
  explicit Polynomial(hid_t dset);

  //! Evaluate the polynomials
  //! \param[in] x independent variable
  //! \return Polynomial evaluated at x
  double operator()(double x) const;
// private:
  std::vector<double> coef_; //!< Polynomial coefficients
};

struct Polynomial_ {
  rtBufferId<double, 1> coef_;  //!< Polynomial coefficients
  unsigned long num_coeffs;

  __device__ __forceinline__ Polynomial_() {}

  __device__ __forceinline__ Polynomial_(rtBufferId<double, 1> coef_, unsigned long num_coeffs) {
    this->coef_ = coef_;
    this->num_coeffs = num_coeffs;
  }
};

//==============================================================================
//! One-dimensional interpolable function
//==============================================================================

class Tabulated1D : public Function1D {
public:
  Tabulated1D() = default;

  //! Construct function from HDF5 data
  //! \param[in] dset Dataset containing tabulated data
  explicit Tabulated1D(hid_t dset);

  //! Evaluate the tabulated function
  //! \param[in] x independent variable
  //! \return Function evaluated at x
  double operator()(double x) const;

  // Accessors
  const std::vector<double>& x() const { return x_; }
  const std::vector<double>& y() const { return y_; }
// private:
  std::size_t n_regions_ {0}; //!< number of interpolation regions
  std::vector<int> nbt_; //!< values separating interpolation regions
  std::vector<Interpolation> int_; //!< interpolation schemes
  std::size_t n_pairs_; //!< number of (x,y) pairs
  std::vector<double> x_; //!< values of abscissa
  std::vector<double> y_; //!< values of ordinate
};

struct Tabulated1D_ {

  size_t n_regions_; //!< number of interpolation regions
  rtBufferId<int, 1> nbt_; //!< values separating interpolation regions
  rtBufferId<Interpolation, 1> int_; //!< interpolation schemes
  size_t n_pairs_; //!< number of (x,y) pairs
  rtBufferId<double, 1> x_; //!< values of abscissa
  unsigned long x_size;
  rtBufferId<double, 1> y_; //!< values of ordinate

  __device__ __forceinline__ Tabulated1D_() {}

  __device__ __forceinline__ Tabulated1D_(const Tabulated1D &t,
                                          rtBufferId<int, 1> nbt_,
                                          rtBufferId<Interpolation, 1> int_,
                                          rtBufferId<double, 1> x_,
                                          unsigned long x_size,
                                          rtBufferId<double, 1> y_) {
    n_regions_ = t.n_regions_;
    n_pairs_ = t.n_pairs_;
    this->nbt_ = nbt_;
    this->int_ = int_;
    this->x_ = x_;
    this->x_size = x_size;
    this->y_ = y_;
  }
};

//==============================================================================
//! Coherent elastic scattering data from a crystalline material
//==============================================================================

class CoherentElasticXS : public Function1D {
  explicit CoherentElasticXS(hid_t dset);
  double operator()(double E) const;
private:
  std::vector<double> bragg_edges_; //!< Bragg edges in [eV]
  std::vector<double> factors_;     //!< Partial sums of structure factors [eV-b]
};

//! Read 1D function from HDF5 dataset
//! \param[in] group HDF5 group containing dataset
//! \param[in] name Name of dataset
//! \return Unique pointer to 1D function
std::unique_ptr<Function1D> read_function(hid_t group, const char* name);

} // namespace openmc

#endif // OPENMC_ENDF_H
