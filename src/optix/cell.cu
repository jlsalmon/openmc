#include <optix_world.h>

#include "variables.cu"

#include "openmc/particle.h"

using namespace optix;
using namespace openmc;

__forceinline__ __device__
bool _contains(int32_t cell_id, Position_ r, Direction_ u, int32_t on_surface) {
  PerRayData payload = {};

  int num_hits = 0;
  int closest_surface_id = -1;
  float closest_intersection_distance = -1;

  payload.position = make_float3(r.x, r.y, r.z);
  payload.hit = true;

  while (payload.hit) {
    optix::Ray ray(payload.position, make_float3(u.x, u.y, u.z), 0, scene_epsilon);

    // printf(">>> rtTrace(origin=(%f, %f, %f), direction=(%f, %f, %f))\n",
    //        ray.origin.x, ray.origin.y, ray.origin.z,
    //        ray.direction.x, ray.direction.y, ray.direction.z);
    rtTrace(top_object, ray, payload);

    if (payload.hit) {
      num_hits++;

      if (closest_surface_id < 0) { // First closest hit? Store it.
        closest_surface_id = payload.surface_id;
        closest_intersection_distance = payload.intersection_distance;
      }
    }
  }

  // printf("closest_surf: %d, dist: %f, nhits: %d\n",
  //   closest_surface_id, closest_intersection_distance, num_hits);

  bool contains = false;

  // FIXME: support multiple cells. Need to know which surface belongs to which
  //  cell on the device

  if (num_hits == 0) {
    contains = false;
  } else if (cell_id == 1) { // bounding cube
    if (closest_surface_id <= 11) {
      if (num_hits % 2 == 0) {
        contains = false;
      } else {
        contains = true;
      }
    } else {
      if (num_hits % 2 == 0) {
        contains = false;
      } else {
        contains = true;
      }
    }
  }

  else if (cell_id == 2) { // inner object
    if (closest_surface_id <= 11) {
      contains = false;
    } else {
      if (num_hits % 2 == 0) {
        contains = true;
      } else {
        contains = false;
      }
    }
  } else {
    printf(">>> ERROR: Only two cells are currently supported\n");
  }

  // printf(">>> _contains(cell=%d, contains=%s, origin=(%f, %f, %f), direction=(%f, %f, %f))\n",
  //        cell_id, contains? "TRUE": "FALSE", r.x, r.y, r.z,
  //        u.x, u.y, u.z);
  return contains;
};