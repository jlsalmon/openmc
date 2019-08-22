#include <optix_world.h>

#include "variables.cu"
#include "surface.cu"

#include "openmc/particle.h"

using namespace optix;
using namespace openmc;

__forceinline__ __device__
bool _rt_contains(Cell_& c, Position_& r, Direction_& u, int32_t on_surface) {
  PerRayData payload = {};

  int num_hits = 0;
  int closest_surface_id = -1;
  float closest_intersection_distance = -1;

  payload.position = make_float3(r.x, r.y, r.z);
  payload.hit = true;

  while (payload.hit) {
    optix::Ray ray(payload.position, make_float3(u.x, u.y, u.z), 0, scene_epsilon);

    rtPrintf(">>> cell.contains: rtTrace(origin=(%f, %f, %f), direction=(%f, %f, %f), on_surface=%i)\n",
           ray.origin.x, ray.origin.y, ray.origin.z,
           ray.direction.x, ray.direction.y, ray.direction.z, on_surface);
    rtTrace(top_object, ray, payload);

    if (payload.hit) {
      num_hits++;

      if (closest_surface_id < 0) { // First closest hit? Store it.
        closest_surface_id = payload.surface_id;
        closest_intersection_distance = payload.intersection_distance;
      }
    }
  }

  rtPrintf("closest_surf: %d, dist: %f, nhits: %d\n",
    closest_surface_id, closest_intersection_distance, num_hits);

  bool contains = false;

  // FIXME: support multiple cells. Need to know which surface belongs to which
  //  cell on the device using the maps

  if (num_hits == 0) {
    contains = false;
  } else if (c.id_ == 1) { // bounding cube
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

  else if (c.id_ == 2) { // inner object
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

  rtPrintf(">>> _rt_contains(cell=%d, contains=%s, origin=(%f, %f, %f), direction=(%f, %f, %f))\n",
           c.id_, contains? "TRUE": "FALSE", r.x, r.y, r.z, u.x, u.y, u.z);
  return contains;
}

__forceinline__ __device__
std::pair<double, int32_t> _rt_distance(Position_& r, Direction_& u, int32_t on_surface) {
  PerRayData payload = {};
  optix::Ray ray(make_float3(r.x, r.y, r.z), make_float3(u.x, u.y, u.z), 0, scene_epsilon);

  rtPrintf(">>> _rt_distance: rtTrace(origin=(%f, %f, %f), direction=(%f, %f, %f))\n",
           ray.origin.x, ray.origin.y, ray.origin.z,
           ray.direction.x, ray.direction.y, ray.direction.z);
  rtTrace(top_object, ray, payload);

  // printf(">>> _distance_to_boundary(origin=(%f, %f, %f), direction=(%f, %f, %f))\n",
  //        r.x, r.y, r.z, u.x, u.y, u.z);
  return {payload.intersection_distance, payload.surface_id};
}

__forceinline__ __device__
bool _csg_contains_simple(Cell_& c, Position_& r, Direction_& u, int32_t on_surface) {
  rtPrintf("c.rpn_.size(): %d\n", c.rpn_.size());
  for (int i = 0; i < c.rpn_.size(); ++i) {
    int32_t token = c.rpn_[i];
    // Assume that no tokens are operators. Evaluate the sense of particle with
    // respect to the surface and see if the token matches the sense. If the
    // particle's surface attribute is set and matches the token, that
    // overrides the determination based on sense().
    if (token == on_surface) {
    } else if (-token == on_surface) {
      return false;
    } else {
      // Note the off-by-one indexing
      Surface_& surf = surface_buffer[abs(token)-1];
      bool sense = _sense(surf, r, u);
      if (sense != (token > 0)) {return false;}
    }
  }
  return true;
}

__forceinline__ __device__
bool _csg_contains_complex(Cell_& c, Position_& r, Direction_& u, int32_t on_surface) {
  // Make a stack of booleans.  We don't know how big it needs to be, but we do
  // know that rpn.size() is an upper-bound.
  bool *stack = (bool *) malloc(c.rpn_.size());
  int i_stack = -1;

  for (int i = 0; i < c.rpn_.size(); ++i) {
    int32_t token = c.rpn_[i];
    // If the token is a binary operator (intersection/union), apply it to
    // the last two items on the stack. If the token is a unary operator
    // (complement), apply it to the last item on the stack.
    if (token == OP_UNION) {
      stack[i_stack-1] = stack[i_stack-1] || stack[i_stack];
      i_stack --;
    } else if (token == OP_INTERSECTION) {
      stack[i_stack-1] = stack[i_stack-1] && stack[i_stack];
      i_stack --;
    } else if (token == OP_COMPLEMENT) {
      stack[i_stack] = !stack[i_stack];
    } else {
      // If the token is not an operator, evaluate the sense of particle with
      // respect to the surface and see if the token matches the sense. If the
      // particle's surface attribute is set and matches the token, that
      // overrides the determination based on sense().
      i_stack ++;
      if (token == on_surface) {
        stack[i_stack] = true;
      } else if (-token == on_surface) {
        stack[i_stack] = false;
      } else {
        // Note the off-by-one indexing
        Surface_& surf = surface_buffer[abs(token)-1];
        bool sense = _sense(surf, r, u);
        stack[i_stack] = (sense == (token > 0));
      }
    }
  }

  if (i_stack == 0) {
    // The one remaining bool on the stack indicates whether the particle is
    // in the cell.
    return stack[i_stack];
  } else {
    // This case occurs if there is no region specification since i_stack will
    // still be -1.
    return true;
  }
}

__forceinline__ __device__
bool _csg_contains(Cell_& c, Position_& r, Direction_& u, int32_t on_surface) {
  bool contains;
  if (c.simple_) {
    contains = _csg_contains_simple(c, r, u, on_surface);
  } else {
    contains = _csg_contains_complex(c, r, u, on_surface);
  }

  rtPrintf(">>> _csg_contains(cell=%d, contains=%s, origin=(%f, %f, %f), direction=(%f, %f, %f))\n",
           c.id_, contains? "TRUE": "FALSE", r.x, r.y, r.z, u.x, u.y, u.z);
  return contains;
}

__forceinline__ __device__
std::pair<float, int32_t> _csg_distance(Cell_& c, Position_& r, Direction_& u, int32_t on_surface) {
  float min_dist = (float) 1e10; //{std::numeric_limits<float>::max()};
  int32_t i_surf {std::numeric_limits<int32_t>::max()};

  for (int i = 0; i < c.rpn_.size(); ++i) {
    int32_t token = c.rpn_[i];

    // Ignore this token if it corresponds to an operator rather than a region.
    if (token >= OP_UNION) continue;

    // Calculate the distance to this surface.
    // Note the off-by-one indexing
    bool coincident {std::abs(token) == std::abs(on_surface)};
    Surface_& surf = surface_buffer[abs(token)-1];
    float d {_distance(surf, r, u, coincident)};
    rtPrintf("d=%f, min_dist=%.20f\n", d, min_dist);
    // Check if this distance is the new minimum.
    if (d < min_dist) {
      rtPrintf("d < min_dist\n");
      rtPrintf("fabsf(d - min_dist) = %.20f\n", fabsf(d - min_dist));
      rtPrintf("fabsf(d - min_dist) / min_dist = %.20f\n", fabsf(d - min_dist) / min_dist);
      rtPrintf("(float) FP_PRECISION = %.20f\n", (float) FP_PRECISION);
      if (fabsf(d - min_dist) / min_dist >= (float) FP_PRECISION) {
        min_dist = d;
        i_surf = -token;
        rtPrintf("i_surf=%d\n", i_surf);
      }
    }
  }

  rtPrintf(">>> _rt_distance: (origin=(%f, %f, %f), direction=(%f, %f, %f)), min_dist=%f, i_surf=%d\n",
           r.x, r.y, r.z, u.x, u.y, u.z, min_dist, i_surf);

  return {min_dist, i_surf};
}