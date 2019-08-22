#pragma once

#include <optix_world.h>

#include "variables.cu"
#include "geometry.cu"
#include "material.cu"
#include "physics.cu"

#include "openmc/particle.h"

using namespace optix;
using namespace openmc;


__device__ __forceinline__
void _cross_surface(Particle_ &p)
{
  int i_surface = abs(p.surface_);
  // TODO: off-by-one
  // const auto& surf {model::surfaces[i_surface - 1].get()}; // FIXME: access to surface instances
  int surface_id = i_surface;
  int boundary_cond;
  if (surface_id <= 11) { // FIXME: flexible boundary conditions
    boundary_cond = _BC_VACUUM;
  } else {
    boundary_cond = _BC_TRANSMIT;
  }
  // if (settings::verbosity >= 10 || simulation::trace) {
  //   write_message("    Crossing surface " + std::to_string(surf->id_));
  // }
  rtPrintf("Crossing surface %i\n", surface_id);

  if (boundary_cond /*surf->bc_*/ == _BC_VACUUM /*&& (settings::run_mode != RUN_MODE_PLOTTING)*/) {
    // =======================================================================
    // PARTICLE LEAKS OUT OF PROBLEM

    // Kill particle
    p.alive_ = false;

    // Score any surface current tallies -- note that the particle is moved
    // forward slightly so that if the mesh boundary is on the surface, it is
    // still processed

    // if (!model::active_meshsurf_tallies.empty()) { // FIXME: tallies
    //   // TODO: Find a better solution to score surface currents than
    //   // physically moving the particle forward slightly
    //
    //   p.r() += TINY_BIT * p.u();
    //   score_surface_tally(this, model::active_meshsurf_tallies);
    // }

    // Score to global leakage tally
    global_tally_leakage_buffer[launch_index] += p.wgt_;

    // Display message
    // if (settings::verbosity >= 10 || simulation::trace) {
    //   write_message("    Leaked out of surface " + std::to_string(surf->id_));
    // }
    rtPrintf("Leaked out of surface %i\n", surface_id);
    return;

  } else if (boundary_cond /*surf->bc_*/ == _BC_REFLECT/* && (settings::run_mode != RUN_MODE_PLOTTING)*/) {
    // =======================================================================
    // PARTICLE REFLECTS FROM SURFACE

    // // Do not handle reflective boundary conditions on lower universes // FIXME: reflective boundary
    // if (p.n_coord_ != 1) {
    //   // this->mark_as_lost("Cannot reflect particle " + std::to_string(id_) + // FIXME: mark as lost
    //   //                    " off surface in a lower universe.");
    //   printf("Cannot reflect particle %lli off surface in a lower universe.\n", p.id_);
    //   return;
    // }
    //
    // // Score surface currents since reflection causes the direction of the
    // // particle to change. For surface filters, we need to score the tallies
    // // twice, once before the particle's surface attribute has changed and
    // // once after. For mesh surface filters, we need to artificially move
    // // the particle slightly back in case the surface crossing is coincident
    // // with a mesh boundary
    //
    // // if (!model::active_surface_tallies.empty()) { // FIXME: tallies
    // //   score_surface_tally(this, model::active_surface_tallies);
    // // }
    //
    //
    // // if (!model::active_meshsurf_tallies.empty()) { // FIXME: tallies
    // //   Position r {this->r()};
    // //   this->r() -= TINY_BIT * this->u();
    // //   score_surface_tally(this, model::active_meshsurf_tallies);
    // //   this->r() = r;
    // // }
    //
    // // Reflect particle off surface
    // Direction u = surf->reflect(p.r(), p.u());
    //
    // // Make sure new particle direction is normalized
    // p.u() = u / u.norm();
    //
    // // Reassign particle's cell and surface
    // p.coord__[0].cell = p.cell_last__[p.n_coord_last_ - 1];
    // p.surface_ = -p.surface_;
    //
    // // If a reflective surface is coincident with a lattice or universe
    // // boundary, it is necessary to redetermine the particle's coordinates in
    // // the lower universes.
    //
    // p.n_coord_ = 1;
    // if (!_find_cell(p, true)) {
    //   // this->mark_as_lost("Couldn't find particle after reflecting from surface "  // FIXME: mark as lost
    //   //                    + std::to_string(surf->id_) + ".");
    //   printf("Couldn't find particle after reflecting from surface %i.\n", surf->id_);
    //   return;
    // }
    //
    // // Set previous coordinate going slightly past surface crossing
    // p.r_last_current_ = p.r() + TINY_BIT*p.u();
    //
    // // Diagnostic message
    // // if (settings::verbosity >= 10 || simulation::trace) {
    //   // write_message("    Reflected from surface " + std::to_string(surf->id_));
    //   printf("Reflected from surface %i.\n", surf->id_);
    // // }
    // return;

  } else if (boundary_cond /*surf->bc_*/ == _BC_PERIODIC /*&& settings::run_mode != RUN_MODE_PLOTTING*/) {
    // =======================================================================
    // PERIODIC BOUNDARY

    // // Do not handle periodic boundary conditions on lower universes // FIXME: periodic boundary
    // if (n_coord_ != 1) {
    //   this->mark_as_lost("Cannot transfer particle " + std::to_string(id_) +
    //                      " across surface in a lower universe. Boundary conditions must be "
    //                      "applied to root universe.");
    //   return;
    // }
    //
    // // Score surface currents since reflection causes the direction of the
    // // particle to change -- artificially move the particle slightly back in
    // // case the surface crossing is coincident with a mesh boundary
    // if (!model::active_meshsurf_tallies.empty()) {
    //   Position r {this->r()};
    //   this->r() -= TINY_BIT * this->u();
    //   score_surface_tally(this, model::active_meshsurf_tallies);
    //   this->r() = r;
    // }
    //
    // // Get a pointer to the partner periodic surface
    // auto surf_p = dynamic_cast<PeriodicSurface*>(surf);
    // auto other = dynamic_cast<PeriodicSurface*>(
    //   model::surfaces[surf_p->i_periodic_].get());
    //
    // // Adjust the particle's location and direction.
    // bool rotational = other->periodic_translate(surf_p, this->r(), this->u());
    //
    // // Reassign particle's surface
    // // TODO: off-by-one
    // surface_ = rotational ?
    //            surf_p->i_periodic_ + 1 :
    //            std::copysign(surf_p->i_periodic_ + 1, surface_);
    //
    // // Figure out what cell particle is in now
    // n_coord_ = 1;
    //
    // if (!find_cell(this, true)) {
    //   this->mark_as_lost("Couldn't find particle after hitting periodic "
    //                      "boundary on surface " + std::to_string(surf->id_) + ".");
    //   return;
    // }
    //
    // // Set previous coordinate going slightly past surface crossing
    // r_last_current_ = this->r() + TINY_BIT*this->u();
    //
    // // Diagnostic message
    // if (settings::verbosity >= 10 || simulation::trace) {
    //   write_message("    Hit periodic boundary on surface " +
    //                 std::to_string(surf->id_));
    // }
    // return;
  }

  // ==========================================================================
  // SEARCH NEIGHBOR LISTS FOR NEXT CELL

// #ifdef DAGMC
//   if (settings::dagmc) {
//     auto cellp = dynamic_cast<DAGCell*>(model::cells[cell_last_[0]].get());
//     // TODO: off-by-one
//     auto surfp = dynamic_cast<DAGSurface*>(model::surfaces[std::abs(surface_) - 1].get());
//     int32_t i_cell = next_cell(cellp, surfp) - 1;
//     // save material and temp
//     material_last_ = material_;
//     sqrtkT_last_ = sqrtkT_;
//     // set new cell value
//     coord_[0].cell = i_cell;
//     cell_instance_ = 0;
//     material_ = model::cells[i_cell]->material_[0];
//     sqrtkT_ = model::cells[i_cell]->sqrtkT_[0];
//     return;
//   }
// #endif

  if (_find_cell(p, true)) return;

  // ==========================================================================
  // COULDN'T FIND PARTICLE IN NEIGHBORING CELLS, SEARCH ALL CELLS
  rtPrintf("Couldn't find particle in neighbouring cells, searching all cells (%d)\n", launch_index);

  // Remove lower coordinate levels and assignment of surface
  p.surface_ = 0;
  p.n_coord_ = 1;
  bool found = _find_cell(p, false);

  if (/*settings::run_mode != RUN_MODE_PLOTTING && */(!found)) {
    // If a cell is still not found, there are two possible causes: 1) there is
    // a void in the model, and 2) the particle hit a surface at a tangent. If
    // the particle is really traveling tangent to a surface, if we move it
    // forward a tiny bit it should fix the problem.

    p.n_coord_ = 1;
    p.r() += (float) TINY_BIT * p.u();

    // Couldn't find next cell anywhere! This probably means there is an actual
    // undefined region in the geometry.

    if (!_find_cell(p, false)) {
      // this->mark_as_lost("After particle " + std::to_string(id_) +  // FIXME: mark as lost
      //       //                    " crossed surface " + std::to_string(surf->id_) +
      //       //                    " it could not be located in any cell and it did not leak.");
      rtPrintf("After particle %lli crossed surface %i it could not be located in "
             "any cell and it did not leak. (%d)\n", p.id_, boundary_cond, launch_index);
      return;
    }
  }
}

__device__ __forceinline__
void _from_source(Particle_ &p, const Particle_::Bank_* src)
{
  // reset some attributes
  p.clear();
  p.alive_ = true;
  p.surface_ = 0;
  p.cell_born_ = C_NONE;
  p.material_ = C_NONE;
  p.n_collision_ = 0;
  p.fission_ = false;

  // copy attributes from source bank site
  p.type_ = src->particle;
  p.wgt_ = src->wgt;
  p.wgt_last_ = src->wgt;
  p.r() = src->r;
  p.u() = src->u;
  p.r_last_current_ = src->r;
  p.r_last_ = src->r;
  p.u_last_ = src->u;
  /*if (settings::run_CE) {*/ // FIXME: multigroup
  p.E_ = src->E;
  rtPrintf("p.E_: %lf\n", p.E_);
  p.g_ = 0;
/*  } else {
    p.g_ = static_cast<int>(src->E);
    p.g_last_ = static_cast<int>(src->E);
    p.E_ = data::energy_bin_avg[g_ - 1];
  }*/
  p.E_last_ = p.E_;
}


__device__ __forceinline__
void _transport(Particle_ &p) {
  // Display message if high verbosity or trace is on
  // if (settings::verbosity >= 9 || simulation::trace) {
  //   write_message("Simulating Particle " + std::to_string(id_));
  // }
  // printf("Simulating particle %lli\n", p.id_);

  // Initialize number of events to zero
  int n_event = 0;

  // Add paricle's starting weight to count for normalizing tallies later
// #pragma omp atomic
  total_weight_buffer[launch_index] += p.wgt_;

  // Force calculation of cross-sections by setting last energy to zero
  // if (settings::run_CE) { // FIXME: add settings as variables
  // for (auto& micro : neutron_xs_) micro.last_E = 0.0;
  for (int i = 0; i < num_nuclides /*data::nuclides.size()*/; ++i) {  // FIXME: copy data::nuclides as buffer
    p.neutron_xs_[i].last_E = 0.0f;
  }
  // }

  // Prepare to write out particle track.
  // if (write_track_) add_particle_track(); // FIXME: support writing tracks

  // Every particle starts with no accumulated flux derivative.
  // if (!model::active_tallies.empty()) zero_flux_derivs(); // FIXME: flux derivative

  while (true) {

    // Set the random number stream
    if (p.type_ == Particle::Type::neutron) { // FIXME: random number generation?
      prn_set_stream(_STREAM_TRACKING);
    } else {
      prn_set_stream(_STREAM_PHOTON);
    }

    // Store pre-collision particle properties
    p.wgt_last_ = p.wgt_;
    p.E_last_ = p.E_;
    p.u_last_ = p.u();
    p.r_last_ = p.r();

    if (p.u().z == 0.f) { //F IXME: this is necessary for the buddha model.. but why?
      p.u() = {0.f, 0.f, 1.f};
    }

    // If the cell hasn't been determined based on the particle's location,
    // initiate a search for the current cell. This generally happens at the
    // beginning of the history and again for any secondary particles
    if (p.coord_[p.n_coord_ - 1].cell == C_NONE) {
      if (!_find_cell(p, false)) {
        printf("Lost particle!\n");

        // this->mark_as_lost("Could not find the cell containing particle " // FIXME: mark as lost
        //                    + std::to_string(id_));
        return;
      }

      // printf("Found it!\n");

      // set birth cell attribute
      if (p.cell_born_ == C_NONE) p.cell_born_ = p.coord_[p.n_coord_ - 1].cell;
    }

    //   // Write particle track.
    //   if (write_track_) write_particle_track(*this); // FIXME: particle tracks

    //   if (settings::check_overlaps) check_cell_overlap(this); // FIXME

    // printf("p.material: %d\n", p.material_);

    // Calculate microscopic and macroscopic cross sections
    if (p.material_ != MATERIAL_VOID) {
      // printf("mat: %d last: %d sqrtkT: %lf last: %lf\n", p.material_, p.material_last_, p.sqrtkT_, p.sqrtkT_last_);

      // if (settings::run_CE) { // FIXME: multigroup
      if (p.material_ != p.material_last_ || p.sqrtkT_ != p.sqrtkT_last_) {
        // If the material is the same as the last material and the
        // temperature hasn't changed, we don't need to lookup cross
        // sections again.
        // model::materials[p.material_]->calculate_xs(*this); // FIXME: cross sections
        _calculate_xs(p);
      }
      // } else {
      //   // Get the MG data
      //   calculate_xs_c(material_, g_, sqrtkT_, this->u_local(),
      //                  macro_xs_.total, macro_xs_.absorption, macro_xs_.nu_fission);
      //
      //   // Finally, update the particle group while we have already checked
      //   // for if multi-group
      //   g_last_ = g_;
      // }
    } else {
      p.macro_xs_.total      = 0.0f;
      p.macro_xs_.absorption = 0.0f;
      p.macro_xs_.fission    = 0.0f;
      p.macro_xs_.nu_fission = 0.0f;
    }

    // Find the distance to the nearest boundary
    auto boundary = _distance_to_boundary(p);

    // printf("BoundaryInfo: dist=%f, id=%d\n", boundary.distance, boundary.surface_index);
    // printf("p.macro_xs_.total: %lf\n", p.macro_xs_.total);
    // printf("p.macro_xs_.absorption: %lf\n", p.macro_xs_.absorption);
    // printf("p.macro_xs_.fission: %lf\n", p.macro_xs_.fission);
    // printf("p.macro_xs_.nu_fission: %lf\n", p.macro_xs_.nu_fission);

    // Sample a distance to collision
    float d_collision;
    if (p.type_ == Particle::Type::electron ||
        p.type_ == Particle::Type::positron) {
      d_collision = 0.0f;
    } else if (p.macro_xs_.total == 0.0f) {
      d_collision = INFINITY;
    } else {
      d_collision = -logf(prn()) / p.macro_xs_.total;
    }

    // Select smaller of the two distances
    float distance = fminf(boundary.distance, d_collision);

    rtPrintf("boundary.distance=%f\n", boundary.distance);
    rtPrintf("d_collision=%f\n", d_collision);

    // Advance particle
    for (int j = 0; j < p.n_coord_; ++j) {
      p.coord_[j].r += distance * p.coord_[j].u;
    }

    //   // Score track-length tallies // FIXME: tallies
    //   if (!model::active_tracklength_tallies.empty()) {
    //     score_tracklength_tally(this, distance);
    //   }
    //
    // Score track-length estimate of k-eff
    // if (settings::run_mode == RUN_MODE_EIGENVALUE && // FIXME: run modes
    //     type_ == Particle::Type::neutron) { // FIXME: particle types
      global_tally_tracklength_buffer[launch_index]
        += p.wgt_ * distance * p.macro_xs_.nu_fission;
    // }
    //
    //   // Score flux derivative accumulators for differential tallies.
    //   if (!model::active_tallies.empty()) {
    //     score_track_derivative(this, distance);
    //   }

    if (d_collision > boundary.distance) {
      // ====================================================================
      // PARTICLE CROSSES SURFACE

      // Set surface that particle is on and adjust coordinate levels
      p.surface_ = boundary.surface_index;
      p.n_coord_ = boundary.coord_level;

      // Saving previous cell data
      for (int j = 0; j < p.n_coord_; ++j) {
        p.cell_last_[j] = p.coord_[j].cell;
      }
      p.n_coord_last_ = p.n_coord_;

      //     if (boundary.lattice_translation[0] != 0 || // FIXME: lattice
      //         boundary.lattice_translation[1] != 0 ||
      //         boundary.lattice_translation[2] != 0) {
      //       // Particle crosses lattice boundary
      //       cross_lattice(this, boundary);
      //       event_ = EVENT_LATTICE;
      //     } else {
      // Particle crosses surface
      _cross_surface(p);
      p.event_ = EVENT_SURFACE;
      //     }

      //     // Score cell to cell partial currents //  FIXME: tallies
      //     if (!model::active_surface_tallies.empty()) {
      //       score_surface_tally(this, model::active_surface_tallies);
      //     }
    } else {
      // ====================================================================
      // PARTICLE HAS COLLISION

      // Score collision estimate of keff
      // if (settings::run_mode == RUN_MODE_EIGENVALUE && // FIXME: run modes
      //     type_ == Particle::Type::neutron) { // FIXME: particle types
        global_tally_collision_buffer[launch_index] += p.wgt_ * p.macro_xs_.nu_fission
                                  / p.macro_xs_.total;
      // }

      //     // Score surface current tallies -- this has to be done before the collision
      //     // since the direction of the particle will change and we need to use the
      //     // pre-collision direction to figure out what mesh surfaces were crossed
      //
      //     if (!model::active_meshsurf_tallies.empty())
      //       score_surface_tally(this, model::active_meshsurf_tallies);

      // Clear surface component
      p.surface_ = 0;

      //     if (settings::run_CE) {
      _collision(p);
      //     } else {
      //       collision_mg(this);
      //     }

      //     // Score collision estimator tallies -- this is done after a collision
      //     // has occurred rather than before because we need information on the
      //     // outgoing energy for any tallies with an outgoing energy filter
      //     if (!model::active_collision_tallies.empty()) score_collision_tally(this); // FIXME: tallies
      //     if (!model::active_analog_tallies.empty()) {
      //       if (settings::run_CE) {
      //         score_analog_tally_ce(this);
      //       } else {
      //         score_analog_tally_mg(this);
      //       }
      //     }

      // Reset banked weight during collision
      p.n_bank_ = 0;
      p.wgt_bank_ = 0.0f;
      for (int& v : p.n_delayed_bank_) v = 0;

      // Reset fission logical
      p.fission_ = false;

      // Save coordinates for tallying purposes
      p.r_last_current_ = p.r();

      // Set last material to none since cross sections will need to be
      // re-evaluated
      p.material_last_ = C_NONE;

      // Set all directions to base level -- right now, after a collision, only
      // the base level directions are changed
      for (int j = 0; j < p.n_coord_ - 1; ++j) {
        // if (p.coord__[j + 1].rotated) { // FIXME
        //   // If next level is rotated, apply rotation matrix
        //   const auto& m {model::cells[p.coord__[j].cell]->rotation_};
        //   const auto& u {p.coord__[j].u};
        //   coord_[j + 1].u.x = m[3]*u.x + m[4]*u.y + m[5]*u.z;
        //   coord_[j + 1].u.y = m[6]*u.x + m[7]*u.y + m[8]*u.z;
        //   coord_[j + 1].u.z = m[9]*u.x + m[10]*u.y + m[11]*u.z;
        // } else {
        // Otherwise, copy this level's direction
        p.coord_[j+1].u = p.coord_[j].u;
        // }
      }

      // // Score flux derivative accumulators for differential tallies. // FIXME: tallies
      // if (!model::active_tallies.empty()) score_collision_derivative(this);
    }

    // If particle has too many events, display warning and kill it
    ++n_event;
    if (n_event == MAX_EVENTS) {
      // warning("Particle " + std::to_string(id_) +
      //         " underwent maximum number of events.");
      printf("Particle %lli underwent maximum number of events at index %d.\n", p.id_, launch_index);
      p.alive_ = false;
    }

    // Check for secondary particles if this particle is dead
    if (!p.alive_) {
      break; // // If no secondary particles, break out of event loop // FIXME: secondary particles
      // if (simulation::secondary_bank.empty()) break;
      //
      // _from_source(&simulation::secondary_bank.back());
      // simulation::secondary_bank.pop_back();
      // n_event = 0;
      //
      // // Enter new particle in particle track file
      // if (write_track_) add_particle_track();
    }
  }

  // // Finish particle track output.
  // if (write_track_) {
  //   write_particle_track(*this);
  //   finalize_particle_track(*this);
  // }
}