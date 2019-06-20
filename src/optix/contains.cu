#include <optix_world.h>
#include <optix_cuda.h>

#include "common.h"
#include "helpers.h"

using namespace optix;

struct PerRayData_contains {
  int num_hits;
};

rtDeclareVariable(float3, ray_origin, ,);
rtDeclareVariable(float3, ray_direction, ,);
rtDeclareVariable(float, scene_epsilon, ,);
rtDeclareVariable(rtObject, top_object, ,);
rtDeclareVariable(uint2, launch_index, rtLaunchIndex,);
rtDeclareVariable(PerRayData_contains, prd, rtPayload,);

rtBuffer<float, 1> output_contains_buffer;

RT_PROGRAM void generate_ray() {
  optix::Ray ray(ray_origin, ray_direction, 0, scene_epsilon);
  PerRayData_contains payload = {};

  rtTrace(top_object, ray, payload);

  output_contains_buffer[0] = payload.num_hits;

  printf(">>> contains.generate_ray() { num_hits=%d }\n", payload.num_hits);
}

RT_PROGRAM void any_hit() {
  prd.num_hits++;
  printf(">>> contains.any_hit() { num_hits=%d }\n", prd.num_hits);
}

RT_PROGRAM void closest_hit() {
  printf(">>> contains.closest_hit() { }\n");
}

RT_PROGRAM void miss() {
  printf(">>> contains.miss()\n");
}

RT_PROGRAM void exception() {
  printf(">>> contains.exception() { 0x%X at launch index (%d,%d) }\n",
         rtGetExceptionCode(), launch_index.x, launch_index.y);
}