#include <optix_world.h>
#include <optix_cuda.h>

#include "variables.cu"
#include "geometry.cu"
#include "math_functions.cu"

#include "openmc/particle.h"
#include "openmc/cell.h"

using namespace optix;
using namespace openmc;


RT_PROGRAM void sample_source() {
  // initialize random number seed
  int64_t id = launch_index + 1;
  set_particle_seed(id);
  prn_set_stream(_STREAM_SOURCE);

  Particle::Type particle_ = Particle::Type::neutron;
  Particle::Bank site;

  // Set weight to one by default
  site.wgt = 1.0;

  // Repeat sampling source location until a good site has been found
  bool found = false;
  int n_reject = 0;
  static int n_accept = 0;
  while (!found) {
    // Set particle type
    site.particle = particle_;

    // Sample spatial distribution
    site.r = Position{0, 0, 0}; // FIXME: space_->sample();
    double xyz[] {site.r.x, site.r.y, site.r.z};

    // Now search to see if location exists in geometry
    int32_t cell_index, instance;
    int err = _openmc_find_cell(xyz, &cell_index, &instance);
    found = (err != OPENMC_E_GEOMETRY_);

    //   // Check if spatial site is in fissionable material
    //   if (found) {
    //     auto space_box = dynamic_cast<SpatialBox*>(space_.get());
    //     if (space_box) {
    //       if (space_box->only_fissionable()) {
    //         // Determine material
    //         const auto& c = model::cells[cell_index];
    //         auto mat_index = c->material_.size() == 1
    //                          ? c->material_[0] : c->material_[instance];
    //
    //         if (mat_index == MATERIAL_VOID) {
    //           found = false;
    //         } else {
    //           if (!model::materials[mat_index]->fissionable_) found = false;
    //         }
    //       }
    //     }
    //   }
    //
    //   // Check for rejection
    //   if (!found) {
    //     ++n_reject;
    //     if (n_reject >= EXTSRC_REJECT_THRESHOLD &&
    //         static_cast<double>(n_accept)/n_reject <= EXTSRC_REJECT_FRACTION) {
    //       fatal_error("More than 95% of external source sites sampled were "
    //                   "rejected. Please check your external source definition.");
    //     }
    //   }
  }

  // Increment number of accepted samples
  // ++n_accept;

  // Sample angle
  site.u = {0.0, 0.0, 1.0}; // FIXME: angle_->sample();

  // // Check for monoenergetic source above maximum particle energy
  // auto p = static_cast<int>(particle_);
  // auto energy_ptr = dynamic_cast<Discrete*>(energy_.get());
  // if (energy_ptr) {
  //   auto energies = xt::adapt(energy_ptr->x());
  //   if (xt::any(energies > data::energy_max[p])) {
  //     fatal_error("Source energy above range of energies of at least "
  //                 "one cross section table");
  //   } else if (xt::any(energies < data::energy_min[p])) {
  //     fatal_error("Source energy below range of energies of at least "
  //                 "one cross section table");
  //   }
  // }

  // while (true) { FIXME
    // Sample energy spectrum
    site.E = watt_spectrum(0.988e6, 2.249e-6); // FIXME: energy_->sample();
    // printf("site.E: %f\n", site.E);

    // Resample if energy falls outside minimum or maximum particle energy
    // if (site.E < data::energy_max[p] && site.E > data::energy_min[p]) break; FIXME
  // }

  // Set delayed group
  site.delayed_group = 0;

  // Write to output buffer
  source_bank_buffer[launch_index] = site;

  prn_set_stream(_STREAM_TRACKING);
}


RT_PROGRAM void exception() {
  printf(">>> simulation.exception() { 0x%X at launch index (%d) }\n",
         rtGetExceptionCode(), launch_index);
}

