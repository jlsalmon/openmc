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

  // u.x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  // u.y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  // u.z = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
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
    // printf("Outside cube and bunny\n");
  }

  else if (id_ == 1) { // cube

    // If we hit the cube first (first 12 surfaces) then we are either inside
    // or outside it (not inside the bunny)
    if (closest_surface_id <= 11) {
      // printf("Hit the cube first (id=%d)\n", closest_surface_id);

      // If we have an even number of hits, we were outside the cube
      if (num_hits % 2 == 0) {
        // printf("Even number of hits => outside cube\n");
        contains = false;
      } else {
        // printf("Odd number of hits => inside cube\n");
        contains = true;
      }
      // contains = num_hits % 2 != 0;
    }
      // If we hit the bunny first, we were inside the cube
    else {
      // printf("Hit the bunny first (id=%d)\n", closest_surface_id);

      // If we have an even number of hits, we must have been inside the bunny
      if (num_hits % 2 == 0) {
        // printf("Even number of hits => inside bunny\n");
        contains = false;
      } else {
        // printf("Odd number of hits => inside cube\n");
        contains = true;
      }
      // contains = num_hits % 2 == 0;
    }

    // printf("Inside cube: %s\n", contains ? "YES": "NO");
  }

  else if (id_ == 2) { // bunny

    // If we hit the cube first then we are not inside the bunny
    if (closest_surface_id <= 11) {
      // printf("Hit the cube first (id=%d)\n", closest_surface_id);
      contains = false;
    }
      // If we hit the bunny first, then we are either inside or outside it
    else {
      // printf("Hit the bunny first (id=%d)\n", closest_surface_id);

      // If we have an even number of hits, we must have hit the bunny an odd
      // number of times and then hit the box, hence we were inside the bunny
      if (num_hits % 2 == 0) {
        // printf("Even number of hits => inside bunny\n");
        contains = true;
      } else {
        // printf("Odd number of hits => inside cube\n");
        contains = false;
      }
      // contains = num_hits % 2 == 0;
    }

    // printf("Inside bunny: %s\n", contains ? "YES": "NO");

  } else {
    throw std::runtime_error("Only two cells are currently supported");
  }


  // TODO: find cell ID from surface ID

  // TODO: odd num bunny hits + one box hit -> inside bunny
  // TODO: even num bunny hits + one box hit -> inside bunny
  // TODO: no bunny hits + one box hit -> inside box
  // TODO: no bunny hits + two box hits -> outside box
  // TODO: no hits -> outside box

  // printf("$$$ cell.contains(id=%ld, pos=(%f,%f,%f), dir=(%f,%f,%f)) { num_hits=%d }: %d\n",
  //        id_, r.x, r.y, r.z, u.x, u.y, u.z, num_hits, contains);
  return contains;
}

void OptiXCell::to_hdf5(hid_t group_id) const { return; }

}