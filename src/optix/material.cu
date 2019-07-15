#pragma once

#include <optix_world.h>

#include "variables.cu"
#include "nuclide.cu"

#include "openmc/material.h"
#include "openmc/particle.h"
#include "openmc/nuclide.h"

using namespace openmc;


__device__ __forceinline__
void _calculate_neutron_xs(Particle_& p)
{
  // printf("Calculating neutron cross section\n");

  // Find energy index on energy grid
  int neutron = static_cast<int>(Particle::Type::neutron);
  int i_grid = std::log(p.E_/energy_min_neutron)/log_spacing;

  // Determine if this material has S(a,b) tables
  bool check_sab = false; // (m.thermal_tables_.size() > 0); FIXME: support thermal tables

  // Initialize position in i_sab_nuclides
  int j = 0;

  // Add contribution from each nuclide in material
  for (int i = 0; i < 1 /*m.nuclide_.size()*/; ++i) { // FIXME: support more than one nuclide
    // ======================================================================
    // CHECK FOR S(A,B) TABLE

    int i_sab = C_NONE;
    double sab_frac = 0.0;

    // Check if this nuclide matches one of the S(a,b) tables specified.
    // This relies on thermal_tables_ being sorted by .index_nuclide
    // if (check_sab) {
    //   // const auto& sab {m.thermal_tables_[j]}; // FIXME: support thermal tables
    //   const auto &sab = thermal_table;
    //   if (i == sab.index_nuclide) {
    //     // Get index in sab_tables
    //     i_sab = sab.index_table;
    //     sab_frac = sab.fraction;
    //
    //     // If particle energy is greater than the highest energy for the
    //     // S(a,b) table, then don't use the S(a,b) table
    //     if (p.E_ > thermal_scatt_buffer[0].threshold()) i_sab = C_NONE;
    //
    //     // Increment position in thermal_tables_
    //     ++j;
    //
    //     // Don't check for S(a,b) tables if there are no more left
    //     // if (j == m.thermal_tables_.size()) check_sab = false; // FIXME: support more than one material
    //     check_sab = false;
    //   }
    // }

    // ======================================================================
    // CALCULATE MICROSCOPIC CROSS SECTION

    // Determine microscopic cross sections for this nuclide
    int i_nuclide = material.nuclide_[i];

    // Calculate microscopic cross section for this nuclide
    const auto& micro {p.neutron_xs_[i_nuclide]};
    // printf("%lf %lf %lf %lf %d %d %lf %lf\n", p.E_, micro.last_E, p.sqrtkT_, micro.last_sqrtkT, i_sab, micro.index_sab, sab_frac, micro.sab_frac);
    if (p.E_ != micro.last_E
        || p.sqrtkT_ != micro.last_sqrtkT
        || i_sab != micro.index_sab
        || sab_frac != micro.sab_frac) {
      // data::nuclides[i_nuclide]->calculate_xs(i_sab, i_grid, sab_frac, p);
      _calculate_xs(nuclide, i_sab, i_grid, sab_frac, p);
    }

    // ======================================================================
    // ADD TO MACROSCOPIC CROSS SECTION

    // Copy atom density of nuclide in material
    double atom_density = material.atom_density_[i];
    // printf("atom density: %lf\n", atom_density);
    // printf("micro.total: %lf\n", micro.total);
    // printf("micro.absorption: %lf\n", micro.absorption);
    // printf("micro.fission: %lf\n", micro.fission);
    // printf("micro.nu_fission: %lf\n", micro.nu_fission);

    // Add contributions to cross sections
    p.macro_xs_.total += atom_density * micro.total;
    p.macro_xs_.absorption += atom_density * micro.absorption;
    p.macro_xs_.fission += atom_density * micro.fission;
    p.macro_xs_.nu_fission += atom_density * micro.nu_fission;
  }
}

__device__ __forceinline__
void _calculate_xs(Particle_& p)
{
  // Set all material macroscopic cross sections to zero
  p.macro_xs_.total = 0.0;
  p.macro_xs_.absorption = 0.0;
  p.macro_xs_.fission = 0.0;
  p.macro_xs_.nu_fission = 0.0;

  if (p.type_ == Particle::Type::neutron) {
    _calculate_neutron_xs(p);
  } /*else if (p.type_ == Particle::Type::photon) {
    _calculate_photon_xs(p);
  }*/ // FIXME: photon support
}
