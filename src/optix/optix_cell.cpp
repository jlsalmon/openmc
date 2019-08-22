#include <cstdlib>

#include "openmc/error.h"

#include "openmc/optix/optix_cell.h"
#include "openmc/optix/optix_geometry.h"

namespace openmc {

using namespace optix;

OptiXCell::OptiXCell() : Cell{} {};

std::pair<double, int32_t>
OptiXCell::distance(Position r, Direction u, int32_t on_surface) const {
  // Which surface, and how far?
  context["ray_origin"]->setFloat(r.x, r.y, r.z);
  context["ray_direction"]->setFloat(u.x, u.y, u.z);
  context->launch(0, 1);

  auto *instance_id_data = static_cast<int *>(instance_id_buffer->map());
  int instance_id = instance_id_data[0] + 1; // TODO: off-by-one
  instance_id_buffer->unmap();

  auto *intersection_distance_data = static_cast<float *>(intersection_distance_buffer->map());
  float distance = intersection_distance_data[0];
  intersection_distance_buffer->unmap();

  // printf("$$$ cell.distance(pos=(%f,%f,%f), dir=(%f,%f,%f)) {}: { dist=%f, id=%d }\n",
  //   r.x, r.y, r.z, u.x, u.y, u.z, distance, instance_id);
  return {distance, instance_id};
}

bool OptiXCell::contains(Position r, Direction u, int32_t on_surface) const {
  context["ray_origin"]->setFloat(r.x, r.y, r.z);
  context["ray_direction"]->setFloat(u.x, u.y, u.z);

  context->launch(0, 1);

  auto *num_hits_data = static_cast<int *>(num_hits_buffer->map());
  int num_hits = num_hits_data[0];
  num_hits_buffer->unmap();

  auto *instance_id_data = static_cast<int *>(instance_id_buffer->map());
  int closest_surface_id = instance_id_data[0]; // TODO: off-by-one
  instance_id_buffer->unmap();

  bool contains;

  if (num_hits == 0) {
    contains = false;
  }

  else if (id_ == 1) { // cube

    if (closest_surface_id <= 11) {

      if (num_hits % 2 == 0) {
        contains = false;
      } else {
        contains = true;
      }
    }
    else {

      if (num_hits % 2 == 0) {
        contains = false;
      } else {
        contains = true;
      }
    }
  }

  else if (id_ == 2) {
    if (closest_surface_id <= 11) {
      contains = false;
    }else {
      if (num_hits % 2 == 0) {
        contains = true;
      } else {
        contains = false;
      }
    }


  } else {
    throw std::runtime_error("Only two cells are currently supported");
  }

  // TODO: find cell ID from surface ID

  printf("$$$ cell.contains(id=%ld, pos=(%f,%f,%f), dir=(%f,%f,%f)) { num_hits=%d }: %d\n",
         id_, r.x, r.y, r.z, u.x, u.y, u.z, num_hits, contains);
  return contains;
}

void OptiXCell::to_hdf5(hid_t group_id) const { return; }

}