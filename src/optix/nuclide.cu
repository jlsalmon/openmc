#pragma once

#include <optix_world.h>

#include "variables.cu"
#include "random_lcg.cu"
#include "math_functions.cu"
#include "reaction_product.cu"

#include "openmc/particle.h"
#include "openmc/nuclide.h"

using namespace openmc;

#define XS_TOTAL_ 0
#define XS_ABSORPTION_ 1
#define XS_FISSION_ 2
#define XS_NU_FISSION_ 3
#define XS_PHOTON_PROD_ 4


__device__ __forceinline__
void _calculate_elastic_xs(const Nuclide_& n, Particle_& p)
{
  // Get temperature index, grid index, and interpolation factor
  auto& micro {p.neutron_xs_[n.i_nuclide_]};
  int i_temp = micro.index_temp;
  int i_grid = micro.index_grid;
  double f = micro.interp_factor;

  if (i_temp >= 0) {
    const auto& xs = n.reactions_[0].xs_[i_temp].value_;
    micro.elastic = (1.0 - f)*xs[i_grid] + f*xs[i_grid + 1];
  }

  // printf("micro.index_temp: %d\n", micro.index_temp);
  // printf("micro.elastic: %lf\n", micro.elastic);
}


__device__ __forceinline__
double _nu(const Nuclide_& n, double E, ReactionProduct::EmissionMode mode, int group=0)
{
  // printf("_nu\n");

  if (!n.fissionable_) return 0.0;

  switch (mode) {
    case ReactionProduct::EmissionMode::prompt:
      if (n.fission_rx_[0].products_[0].is_polynomial_yield) {
        return _polynomial(n.fission_rx_[0].products_[0].polynomial_yield_, E);
      } else {
        return _tabulated_1d(n.fission_rx_[0].products_[0].tabulated_1d_yield_, E);
      }
      // return (n.fission_rx_[0].products_[0].yield_)(E); // FIXME
    case ReactionProduct::EmissionMode::delayed:
      if (n.n_precursor_ > 0) {
        auto rx = n.fission_rx_[0];
        if (group >= 1 && group < rx.products_.size()) {
          // If delayed group specified, determine yield immediately
          if (rx.products_[group].is_polynomial_yield) {
            return _polynomial(rx.products_[group].polynomial_yield_, E);
          } else {
            return _tabulated_1d(rx.products_[group].tabulated_1d_yield_, E);
          }
          // return (rx.products_[group].yield_)(E); // FIXME
        } else {
          double nu {0.0};

          for (int i = 1; i < rx.products_.size(); ++i) {
            // Skip any non-neutron products
            const auto& product = rx.products_[i];
            if (product.particle_ != Particle::Type::neutron) continue;

            // Evaluate yield
            if (product.emission_mode_ == ReactionProduct::EmissionMode::delayed) {
              if (product.is_polynomial_yield) {
                nu += _polynomial(product.polynomial_yield_, E);
              } else {
                nu += _tabulated_1d(product.tabulated_1d_yield_, E);
              }

              // nu += _tabulated_1d(E);
              // nu += (*product.yield_)(E); // FIXME this should be tabulated 1D, not polynomial
            }
          }
          return nu;
        }
      } else {
        return 0.0;
      }
    case ReactionProduct::EmissionMode::total:
      // if (n.total_nu_) { // FIXME: total nu
      //   return (*total_nu_)(E);
      // } else {
        if (n.fission_rx_[0].products_[0].is_polynomial_yield) {
          return _polynomial(n.fission_rx_[0].products_[0].polynomial_yield_, E);
        } else {
          return _tabulated_1d(n.fission_rx_[0].products_[0].tabulated_1d_yield_, E);
        }

        // return (n.fission_rx_[0].products_[0].yield_)(E); // FIXME
      // }
  }
  // UNREACHABLE();
}


__device__ __forceinline__
void _calculate_xs(Nuclide_& n, int i_sab, int i_log_union, double sab_frac, Particle_& p) {
  auto& micro {p.neutron_xs_[n.i_nuclide_]};

  // printf("Calculating nuclide cross section\n");

  // Initialize cached cross sections to zero
  micro.elastic = CACHE_INVALID;
  micro.thermal = 0.0;
  micro.thermal_elastic = 0.0;

  // Check to see if there is multipole data present at this energy
  bool use_mp = false;
  // if (multipole_) { // FIXME: multipole
  //   use_mp = (p.E_ >= multipole_->E_min_ && p.E_ <= multipole_->E_max_);
  // }
  //
  // // Evaluate multipole or interpolate
  if (use_mp) {
  //   // Call multipole kernel
  //   double sig_s, sig_a, sig_f;
  //   std::tie(sig_s, sig_a, sig_f) = multipole_->evaluate(p.E_, p.sqrtkT_);
  //
  //   micro.total = sig_s + sig_a;
  //   micro.elastic = sig_s;
  //   micro.absorption = sig_a;
  //   micro.fission = sig_f;
  //   micro.nu_fission = fissionable_ ?
  //                      sig_f * this->nu(p.E_, EmissionMode::total) : 0.0;
  //
  //   if (simulation::need_depletion_rx) {
  //     // Only non-zero reaction is (n,gamma)
  //     micro.reaction[0] = sig_a - sig_f;
  //
  //     // Set all other reaction cross sections to zero
  //     for (int i = 1; i < DEPLETION_RX.size(); ++i) {
  //       micro.reaction[i] = 0.0;
  //     }
  //   }
  //
  //   // Ensure these values are set
  //   // Note, the only time either is used is in one of 4 places:
  //   // 1. physics.cpp - scatter - For inelastic scatter.
  //   // 2. physics.cpp - sample_fission - For partial fissions.
  //   // 3. tally.F90 - score_general - For tallying on MTxxx reactions.
  //   // 4. nuclide.cpp - calculate_urr_xs - For unresolved purposes.
  //   // It is worth noting that none of these occur in the resolved
  //   // resonance range, so the value here does not matter.  index_temp is
  //   // set to -1 to force a segfault in case a developer messes up and tries
  //   // to use it with multipole.
  //   micro.index_temp = -1;
  //   micro.index_grid = -1;
  //   micro.interp_factor = 0.0;

  } else {
    // Find the appropriate temperature index.
    double kT = p.sqrtkT_*p.sqrtkT_;
    double f;
    int i_temp = -1;
    switch (temperature_method) {
      case TEMPERATURE_NEAREST:
      {
        double max_diff = INFTY;
        for (int t = 0; t < 1; ++t) {
          double diff = std::abs(n.kTs_[t] - kT);
          if (diff < max_diff) {
            i_temp = t;
            max_diff = diff;
          }
        }
      }
        break;

      case TEMPERATURE_INTERPOLATION:
        // Find temperatures that bound the actual temperature
        for (i_temp = 0; i_temp < 1 - 1; ++i_temp) {
          if (n.kTs_[i_temp] <= kT && kT < n.kTs_[i_temp + 1]) break;
        }

        // Randomly sample between temperature i and i+1
        f = (kT - n.kTs_[i_temp]) / (n.kTs_[i_temp + 1] - n.kTs_[i_temp]);
        if (f > prn()) ++i_temp;
        break;
    }

    // Determine the energy grid index using a logarithmic mapping to
    // reduce the energy range over which a binary search needs to be
    // performed

    const auto& grid {n.grid_[i_temp]};
    const auto& xs {n.xs_[i_temp]};

    // printf("i_temp: %d\n", i_temp);
    // printf("grid.energy_size: %lu\n", grid.energy_size);

    int i_grid;
    if (p.E_ < grid.energy[0]) {
      i_grid = 0;
    } else if (p.E_ > grid.energy[grid.energy_size - 1]) {
      // printf("Here\n");
      i_grid = grid.energy_size - 2;
    } else {
      // printf("Oh noooo :(\n");
      // Determine bounding indices based on which equal log-spaced
      // interval the energy is in
      // printf("grid.grid_index_size: %lu\n", grid.grid_index_size);
      // printf("grid_index_buffer[0]: %i\n", grid_index_buffer[0]);
      // printf("grid_index_buffer[-1]: %i\n", grid_index_buffer[grid.grid_index_size - 1]);
      //
      // printf("i_log_union: %d\n", i_log_union);

      int i_low  = grid.grid_index[i_log_union];
      int i_high = grid.grid_index[i_log_union + 1] + 1;

      // printf("i_low: %d i_high: %d\n", i_low, i_high);

      // Perform binary search over reduced range
      i_grid = _lower_bound(i_low, i_high, grid.energy, p.E_);
    }

    // printf("i_grid: %d\n", i_grid);
    // printf("grid.energy[76399]: %lf\n", grid.energy[76399]);
    // printf("grid.energy[76400]: %lf\n", grid.energy[76400]);

    // check for rare case where two energy points are the same
    if (grid.energy[i_grid] == grid.energy[i_grid + 1]) ++i_grid;

    // calculate interpolation factor
    f = (p.E_ - grid.energy[i_grid]) /
        (grid.energy[i_grid + 1] - grid.energy[i_grid]);

    micro.index_temp = i_temp;
    micro.index_grid = i_grid;
    micro.interp_factor = f;

    // printf("f: %f\n", f);
    // printf("p.E_: %lf\n", p.E_);
    // printf("grid.energy[i_grid]: %lf\n", grid.energy[i_grid]);
    // printf("grid.energy[i_grid + 1]: %lf\n", grid.energy[i_grid + 1]);

    // Calculate microscopic nuclide total cross section
    // micro.total = (1.0 - f)*xs(i_grid, XS_TOTAL)
    //               + f*xs(i_grid + 1, XS_TOTAL);
    micro.total = (1.0 - f)*xs[(5 - XS_TOTAL_)*i_grid]
                  + f*xs[(5 - XS_TOTAL_)*i_grid + 1];

    // printf("xs[(5 - XS_TOTAL_)*i_grid]: %lf\n", xs[(5 - XS_TOTAL_)*i_grid]);
    // printf("xs[(5 - XS_TOTAL_)*i_grid + 1]: %lf\n", xs[(5 - XS_TOTAL_)*i_grid + 1]);
    // printf("xs[0,1,2,3,4,5]: %lf %lf %lf %lf %lf %lf\n",
    //        xs[0], xs[1], xs[2],
    //        xs[3], xs[4], xs[5]);
    // printf("xs[5 * 0,1,2,3,4,5]: %lf %lf %lf %lf %lf %lf\n",
    //        xs[5*0], xs[5*1], xs[5*2],
    //        xs[5*3], xs[5*4], xs[5*5]);
    //
    // printf("nuclide._calculate_xs: micro.total: %lf\n", micro.total);

    // Calculate microscopic nuclide absorption cross section
    // micro.absorption = (1.0 - f)*xs(i_grid, XS_ABSORPTION)
    //                    + f*xs(i_grid + 1, XS_ABSORPTION);
    micro.absorption = (1.0 - f)*xs[(5 - XS_ABSORPTION_)*i_grid]
                  + f*xs[(5 - XS_ABSORPTION_)*i_grid + 1];

    // printf("xs[(5 - XS_ABSORPTION_)*i_grid]: %lf\n", xs[(5 - XS_ABSORPTION_)*i_grid]);
    // printf("xs[(5 - XS_ABSORPTION_)*i_grid + 1]: %lf\n", xs[(5 - XS_ABSORPTION_)*i_grid + 1]);
    // printf("nuclide._calculate_xs: micro.absorption: %lf\n", micro.absorption);

    if (n.fissionable_) {
      // Calculate microscopic nuclide total cross section
      // micro.fission = (1.0 - f)*xs(i_grid, XS_FISSION)
      //                 + f*xs(i_grid + 1, XS_FISSION);
      micro.fission = (1.0 - f)*xs[(5 - XS_FISSION_)*i_grid]
                    + f*xs[(5 - XS_FISSION_)*i_grid + 1];

      // Calculate microscopic nuclide nu-fission cross section
      // micro.nu_fission = (1.0 - f)*xs(i_grid, XS_NU_FISSION)
      //                    + f*xs(i_grid + 1, XS_NU_FISSION);
      micro.nu_fission = (1.0 - f)*xs[(5 - XS_NU_FISSION_)*i_grid]
                    + f*xs[(5 - XS_NU_FISSION_)*i_grid + 1];
    } else {
      micro.fission = 0.0;
      micro.nu_fission = 0.0;
    }

  //   // Calculate microscopic nuclide photon production cross section // FIXME: photon support
  //   micro.photon_prod = (1.0 - f)*xs(i_grid, XS_PHOTON_PROD)
  //                       + f*xs(i_grid + 1, XS_PHOTON_PROD);

  //   // Depletion-related reactions // FIXME: depletion reactions
  //   if (simulation::need_depletion_rx) {
  //     // Initialize all reaction cross sections to zero
  //     for (double& xs_i : micro.reaction) {
  //       xs_i = 0.0;
  //     }
  //
  //     for (int j = 0; j < DEPLETION_RX.size(); ++j) {
  //       // If reaction is present and energy is greater than threshold, set the
  //       // reaction xs appropriately
  //       int i_rx = reaction_index_[DEPLETION_RX[j]];
  //       if (i_rx >= 0) {
  //         const auto& rx = reactions_[i_rx];
  //         const auto& rx_xs = rx->xs_[i_temp].value;
  //
  //         // Physics says that (n,gamma) is not a threshold reaction, so we don't
  //         // need to specifically check its threshold index
  //         if (j == 0) {
  //           micro.reaction[0] = (1.0 - f)*rx_xs[i_grid]
  //                               + f*rx_xs[i_grid + 1];
  //           continue;
  //         }
  //
  //         int threshold = rx->xs_[i_temp].threshold;
  //         if (i_grid >= threshold) {
  //           micro.reaction[j] = (1.0 - f)*rx_xs[i_grid - threshold] +
  //                               f*rx_xs[i_grid - threshold + 1];
  //         } else if (j >= 3) {
  //           // One can show that the the threshold for (n,(x+1)n) is always
  //           // higher than the threshold for (n,xn). Thus, if we are below
  //           // the threshold for, e.g., (n,2n), there is no reason to check
  //           // the threshold for (n,3n) and (n,4n).
  //           break;
  //         }
  //       }
  //     }
  //   }
  }

  // Initialize sab treatment to false
  micro.index_sab = C_NONE;
  micro.sab_frac = 0.0;

  // Initialize URR probability table treatment to false
  micro.use_ptable = false;

  // If there is S(a,b) data for this nuclide, we need to set the sab_scatter
  // and sab_elastic cross sections and correct the total and elastic cross
  // sections.

  // if (i_sab >= 0) this->calculate_sab_xs(i_sab, sab_frac, p); // FIXME
  //
  // // If the particle is in the unresolved resonance range and there are // FIXME
  // // probability tables, we need to determine cross sections from the table
  // if (settings::urr_ptables_on && urr_present_ && !use_mp) {
  //   int n = urr_data_[micro.index_temp].n_energy_;
  //   if ((p.E_ > urr_data_[micro.index_temp].energy_(0)) &&
  //       (p.E_ < urr_data_[micro.index_temp].energy_(n-1))) {
  //     this->calculate_urr_xs(micro.index_temp, p);
  //   }
  // }

  micro.last_E = p.E_;
  micro.last_sqrtkT = p.sqrtkT_;
}