#pragma once

#include <optix_world.h>

#include "variables.cu"
#include "cell.cu"

#include "openmc/particle.h"
#include "openmc/cell.h"
#include "openmc/geometry.h"

using namespace optix;
using namespace openmc;

__forceinline__ __device__
bool _find_cell_inner(Particle_& p, const NeighborList* neighbor_list)
{
  int root_universe = 0; // FIXME
  int num_cells = 2; // FIXME
  int num_materials = 1; // FIXME
  int num_sqrtkT = 1; // FIXME

  // Find which cell of this universe the particle is in.  Use the neighbor list
  // to shorten the search if one was provided.
  bool found = false;
  int32_t i_cell;
  if (neighbor_list) {
    // FIXME
    // for (auto it = neighbor_list->cbegin(); it != neighbor_list->cend(); ++it) {
    //   i_cell = *it;
    //
    //   // Make sure the search cell is in the same universe.
    //   int i_universe = p->coord_[p->n_coord_-1].universe;
    //   if (cell_buffer[i_cell]./*model::cells[i_cell]->*/universe_ != i_universe) continue;
    //
    //   // Check if this cell contains the particle.
    //   Position r {p->r_local()};
    //   Direction u {p->u_local()};
    //   auto surf = p->surface_;
    //   if (cell_buffer[i_cell]./*model::cells[i_cell]->*/contains(r, u, surf)) {
    //     p->coord_[p->n_coord_-1].cell = i_cell;
    //     found = true;
    //     break;
    //   }
    // }

  } else {
    // int i_universe = p->coord_[p->n_coord_-1].universe;
    // const auto& univ {*model::universes[i_universe]};
    // const auto& cells {
    //   !univ.partitioner_
    //   ? model::universes[i_universe]->cells_
    //   : univ.partitioner_->get_cells(p->r_local(), p->u_local())
    // };

    // for (auto it = cells.cbegin(); it != cells.cend(); it++) {
    //   i_cell = *it;
    for (int i = 0; i < num_cells; ++i) {
      i_cell = i;
      Cell_& c {cell_buffer[i_cell]};

      // Make sure the search cell is in the same universe.
      int i_universe = p.coord_[p.n_coord_-1].universe;
      if (c.universe_ /*model::cells[i_cell]->universe_*/ != i_universe) continue;

      // Check if this cell contains the particle.
      Position r {p.r_local()};
      Direction u {p.u_local()};
      auto surf = p.surface_;
      if (/*model::cells[i_cell]->*/_contains(c.id_, r, u, surf)) {
        p.coord_[p.n_coord_-1].cell = i_cell;
        found = true;
        break;
      }
    }
  }

  // FIXME: Announce the cell that the particle is entering.
  // if (found && (settings::verbosity >= 10 || simulation::trace)) {
  //   std::stringstream msg;
  //   msg << "    Entering cell " << model::cells[i_cell]->id_;
  //   write_message(msg, 1);
  // }
  // printf("Entering cell %d\n", cell_buffer[i_cell].id_);

  if (found) {
    // Cell_& c {cell_buffer[i_cell]/**model::cells[i_cell]*/};
    Cell_& c {cell_buffer[i_cell]};
    if (c.type_ == FILL_MATERIAL) {
      //=======================================================================
      //! Found a material cell which means this is the lowest coord level.

      // Find the distribcell instance number.
      if (num_materials /* FIXME c.material_.size()*/ > 1 || num_sqrtkT /*FIXME c.sqrtkT_.size()*/ > 1) {
        int offset = 0;
        for (int i = 0; i < p.n_coord_; i++) {
          // FIXME const auto& c_i {cell_buffer[p.coord_[i].cell] /**model::cells[p.coord_[i].cell]*/};
          const auto& c_i {cell_buffer[p.coord_[i].cell]};
          if (c_i.type_ == FILL_UNIVERSE) {
            offset += c_i.offset_[c.distribcell_index_];
          }

          // FIXME
          // else if (c_i.type_ == FILL_LATTICE) {
          //   auto& lat {*model::lattices[p.coord_[i+1].lattice]};
          //   int i_xyz[3] {p.coord_[i+1].lattice_x,
          //                 p.coord_[i+1].lattice_y,
          //                 p.coord_[i+1].lattice_z};
          //   if (lat.are_valid_indices(i_xyz)) {
          //     offset += lat.offset(c.distribcell_index_, i_xyz);
          //   }
          // }
        }
        p.cell_instance_ = offset;
      } else {
        p.cell_instance_ = 0;
      }

      // Set the material and temperature.
      p.material_last_ = p.material_;
      if (num_materials /* FIXME c.material_.size()*/ > 1) {
        p.material_ = c.material_[p.cell_instance_];
      } else {
        p.material_ = c.material_[0];
      }
      p.sqrtkT_last_ = p.sqrtkT_;
      if (num_sqrtkT /*FIXME c.sqrtkT_.size()*/ > 1) {
        p.sqrtkT_ = c.sqrtkT_[p.cell_instance_];
      } else {
        p.sqrtkT_ = c.sqrtkT_[0];
      }

      return true;

    }

    // FIXME
    // else if (c.type_ == FILL_UNIVERSE) {
    //   //========================================================================
    //   //! Found a lower universe, update this coord level then search the next.
    //
    //   // Set the lower coordinate level universe.
    //   p->coord_[p->n_coord_].universe = c.fill_;
    //
    //   // Set the position and direction.
    //   p->coord_[p->n_coord_].r = p->coord_[p->n_coord_-1].r;
    //   p->coord_[p->n_coord_].u = p->coord_[p->n_coord_-1].u;
    //
    //   // Apply translation.
    //   p->coord_[p->n_coord_].r -= c.translation_;
    //
    //   // Apply rotation.
    //   if (!c.rotation_.empty()) {
    //     Position r = p->coord_[p->n_coord_].r;
    //     p->coord_[p->n_coord_].r.x = r.x*c.rotation_[3] + r.y*c.rotation_[4]
    //                                  + r.z*c.rotation_[5];
    //     p->coord_[p->n_coord_].r.y = r.x*c.rotation_[6] + r.y*c.rotation_[7]
    //                                  + r.z*c.rotation_[8];
    //     p->coord_[p->n_coord_].r.z = r.x*c.rotation_[9] + r.y*c.rotation_[10]
    //                                  + r.z*c.rotation_[11];
    //     Direction u = p->coord_[p->n_coord_].u;
    //     p->coord_[p->n_coord_].u.x = u.x*c.rotation_[3] + u.y*c.rotation_[4]
    //                                  + u.z*c.rotation_[5];
    //     p->coord_[p->n_coord_].u.y = u.x*c.rotation_[6] + u.y*c.rotation_[7]
    //                                  + u.z*c.rotation_[8];
    //     p->coord_[p->n_coord_].u.z = u.x*c.rotation_[9] + u.y*c.rotation_[10]
    //                                  + u.z*c.rotation_[11];
    //     p->coord_[p->n_coord_].rotated = true;
    //   }
    //
    //   // Update the coordinate level and recurse.
    //   ++p->n_coord_;
    //   return find_cell_inner(p, nullptr);
    //
    // } else if (c.type_ == FILL_LATTICE) {
    //   //========================================================================
    //   //! Found a lower lattice, update this coord level then search the next.
    //
    //   Lattice& lat {*model::lattices[c.fill_]};
    //
    //   // Determine lattice indices.
    //   auto i_xyz = lat.get_indices(p->r_local(), p->u_local());
    //
    //   // Store lower level coordinates.
    //   Position r = lat.get_local_position(p->r_local(), i_xyz);
    //   p->coord_[p->n_coord_].r = r;
    //   p->coord_[p->n_coord_].u = p->u_local();
    //
    //   // Set lattice indices.
    //   p->coord_[p->n_coord_].lattice = c.fill_;
    //   p->coord_[p->n_coord_].lattice_x = i_xyz[0];
    //   p->coord_[p->n_coord_].lattice_y = i_xyz[1];
    //   p->coord_[p->n_coord_].lattice_z = i_xyz[2];
    //
    //   // Set the lower coordinate level universe.
    //   if (lat.are_valid_indices(i_xyz)) {
    //     p->coord_[p->n_coord_].universe = lat[i_xyz];
    //   } else {
    //     if (lat.outer_ != NO_OUTER_UNIVERSE) {
    //       p->coord_[p->n_coord_].universe = lat.outer_;
    //     } else {
    //       std::stringstream err_msg;
    //       err_msg << "Particle " << p->id_ << " is outside lattice "
    //               << lat.id_ << " but the lattice has no defined outer "
    //                             "universe.";
    //       warning(err_msg);
    //       return false;
    //     }
    //   }
    //
    //   // Update the coordinate level and recurse.
    //   ++p->n_coord_;
    //   return find_cell_inner(p, nullptr);
    // }
  }

  return found;
}


__host__ __forceinline__ __device__
bool _find_cell(Particle_& p, bool use_neighbor_lists)
{
  int root_universe = 0; // FIXME

  // Determine universe (if not yet set, use root universe).
  int i_universe = p.coord_[p.n_coord_-1].universe;
  if (i_universe == C_NONE) {
    p.coord_[0].universe = root_universe; // model::root_universe;
    p.n_coord_ = 1;
    i_universe = root_universe; //model::root_universe;
  }

  // Reset all the deeper coordinate levels.
  for (int i = p.n_coord_; i < 1 /*FIXME: p->coord_.size()*/; i++) {
    p.coord_[i].reset();
  }

  if (use_neighbor_lists) {
    // Get the cell this particle was in previously.
    auto coord_lvl = p.n_coord_ - 1;
    auto i_cell = p.coord_[coord_lvl].cell;
    // Cell_& c {cell_buffer[i_cell] /**model::cells[i_cell]*/};
    Cell_& c {cell_buffer[i_cell]};

    // Search for the particle in that cell's neighbor list.  Return if we
    // found the particle.
    bool found = _find_cell_inner(p, nullptr /*&c.neighbors_*/);
    if (found) return found;

    // The particle could not be found in the neighbor list.  Try searching all
    // cells in this universe, and update the neighbor list if we find a new
    // neighboring cell.
    // found = _find_cell_inner(p, nullptr); FIXME
    // if (found) c.neighbors_.push_back(p->coord_[coord_lvl].cell); FIXME
    return found;

  } else {
    // Search all cells in this universe for the particle.
    return _find_cell_inner(p, nullptr);
  }
}


__forceinline__ __device__
int _openmc_find_cell(const double* xyz, int32_t* index, int32_t* instance)
{
  Particle_ p;

  p.r() = Position{xyz};
  p.u() = {0.0, 0.0, 1.0};

  if (!_find_cell(p, false)) {
    printf("Could not find cell at position (%lf, %lf, %lf).\n",
           p.r().x, p.r().y, p.r().z);
    // std::stringstream msg;
    // msg << "Could not find cell at position (" << p.r().x << ", " << p.r().y
    //     << ", " << p.r().z << ").";
    // set_errmsg(msg);
    return OPENMC_E_GEOMETRY_;
  }

  *index = p.coord_[p.n_coord_-1].cell;
  *instance = p.cell_instance_;
  return 0;
}


__device__ __forceinline__
BoundaryInfo _distance_to_boundary(Particle_& p)
{
  BoundaryInfo info;
  double d_lat = INFINITY;
  double d_surf = INFINITY;
  int32_t level_surf_cross;
  // std::array<int, 3> level_lat_trans {};

  // Loop over each coordinate level.
  for (int i = 0; i < p.n_coord_; i++) {
    Position r {p.coord_[i].r};
    Direction u {p.coord_[i].u};
    // Cell_& c {*model::cells[p.coord__[i].cell]}; // FIXME: cell objects
    Cell_& c {cell_buffer[i]};

    // Find the oncoming surface in this cell and the distance to it.
    // auto surface_distance = c.distance(r, u, p.surface_);
    PerRayData payload = {};
    optix::Ray ray(make_float3(r.x, r.y, r.z), make_float3(u.x, u.y, u.z), 0, scene_epsilon);
    rtTrace(top_object, ray, payload);

    // printf(">>> _distance_to_boundary(origin=(%f, %f, %f), direction=(%f, %f, %f))\n",
    //        r.x, r.y, r.z, u.x, u.y, u.z);

    d_surf = payload.intersection_distance;
    level_surf_cross = payload.surface_id;

    // Find the distance to the next lattice tile crossing.
    // if (p->coord_[i].lattice != C_NONE) { // FIXME: lattice
    //   auto& lat {*model::lattices[p->coord_[i].lattice]};
    //   std::array<int, 3> i_xyz {p->coord_[i].lattice_x, p->coord_[i].lattice_y,
    //                             p->coord_[i].lattice_z};
    //   //TODO: refactor so both lattice use the same position argument (which
    //   //also means the lat.type attribute can be removed)
    //   std::pair<double, std::array<int, 3>> lattice_distance;
    //   switch (lat.type_) {
    //     case LatticeType::rect:
    //       lattice_distance = lat.distance(r, u, i_xyz);
    //       break;
    //     case LatticeType::hex:
    //       Position r_hex {p->coord_[i-1].r.x, p->coord_[i-1].r.y,
    //                       p->coord_[i].r.z};
    //       lattice_distance = lat.distance(r_hex, u, i_xyz);
    //       break;
    //   }
    //   d_lat = lattice_distance.first;
    //   level_lat_trans = lattice_distance.second;
    //
    //   if (d_lat < 0) {
    //     std::stringstream err_msg;
    //     err_msg << "Particle " << p->id_
    //             << " had a negative distance to a lattice boundary";
    //     p->mark_as_lost(err_msg);
    //   }
    // }

    // If the boundary on this coordinate level is coincident with a boundary on
    // a higher level then we need to make sure that the higher level boundary
    // is selected.  This logic must consider floating point precision.
    double& d = info.distance;
    if (d_surf < d_lat - FP_COINCIDENT) {
      if (d == INFINITY || (d - d_surf)/d >= FP_REL_PRECISION) {
        d = d_surf;

        // If the cell is not simple, it is possible that both the negative and
        // positive half-space were given in the region specification. Thus, we
        // have to explicitly check which half-space the particle would be
        // traveling into if the surface is crossed
        if (c.simple_) {
          info.surface_index = level_surf_cross;
        } else {
          Position r_hit = r + d_surf * u; // FIXME: this normal calculation might be wrong
          // Surface& surf {*model::surfaces[std::abs(level_surf_cross)-1]};
          // Direction norm = surf.normal(r_hit);
          float3 normal = normal_buffer[level_surf_cross];
          Direction norm = {normal.x, normal.y, normal.z};
          if (u.dot(norm) > 0) {
            info.surface_index = std::abs(level_surf_cross);
          } else {
            info.surface_index = -std::abs(level_surf_cross);
          }
        }

        // info.lattice_translation[0] = 0; FIXME: lattice
        // info.lattice_translation[1] = 0;
        // info.lattice_translation[2] = 0;
        info.coord_level = i + 1;
      }
    } else {
      if (d == INFINITY || (d - d_lat)/d >= FP_REL_PRECISION) {
        d = d_lat;
        info.surface_index = 0;
        // info.lattice_translation = level_lat_trans; // FIXME: lattice
        info.coord_level = i + 1;
      }
    }
  }
  return info;
}