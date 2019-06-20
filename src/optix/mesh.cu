#include <optix_world.h>
#include <optix_cuda.h>

#include "common.h"
#include "helpers.h"

using namespace optix;

struct PerRayData_geometry {
  int instance_id;
  float intersection_distance;
  int num_hits;
};

rtDeclareVariable(float3, ray_origin, ,);
rtDeclareVariable(float3, ray_direction, ,);
rtDeclareVariable(float, scene_epsilon, ,);
rtDeclareVariable(rtObject, top_object, ,);
rtDeclareVariable(uint2, launch_index, rtLaunchIndex,);
rtDeclareVariable(PerRayData_geometry, payload, rtPayload,);
rtDeclareVariable(float, intersection_distance, rtIntersectionDistance,);
rtDeclareVariable(float2, barycentrics, attribute rtTriangleBarycentrics,);

rtBuffer<int, 1> instance_id_buffer;
rtBuffer<float, 1> intersection_distance_buffer;
rtBuffer<int, 1> num_hits_buffer;


RT_PROGRAM void generate_ray() {
  // printf(">>> geometry.generate_ray(origin=(%f, %f, %f), direction=(%f, %f, %f))\n",
  //        ray_origin.x, ray_origin.y, ray_origin.z,
  //        ray_direction.x, ray_direction.y, ray_direction.z);

  optix::Ray ray(ray_origin, ray_direction, 0, scene_epsilon);
  PerRayData_geometry payload = {};

  rtTrace(top_object, ray, payload);

  instance_id_buffer[0] = payload.instance_id;
  intersection_distance_buffer[0] = payload.intersection_distance;
  num_hits_buffer[0] = payload.num_hits;

  // printf(">>> geometry.generate_ray() { id=%d, dist=%f }\n",
  //   payload.instance_id, payload.intersection_distance);
}

RT_PROGRAM void any_hit() {
  payload.num_hits++;
  printf(">>> geometry.any_hit() { num_hits=%d }\n", payload.num_hits);
}

RT_PROGRAM void closest_hit() {
  payload.num_hits++;
  payload.intersection_distance = intersection_distance;

#if OPTIX_VERSION / 10000 >= 6
  payload.instance_id = rtGetPrimitiveIndex();
  bool front = rtIsTriangleHitFrontFace();
  bool back = rtIsTriangleHitBackFace();
#else
  payload.instance_id = 0;
  bool front = false;
  bool back = false;
#endif

  // printf(">>> geometry.closest_hit() { id=%d, dist=%f, nhits=%d, front=%d, back=%d }\n",
  //        payload.instance_id, payload.intersection_distance, payload.num_hits, front, back);
}

RT_PROGRAM void miss() {
  printf(">>> geometry.miss()\n");
}

RT_PROGRAM void exception() {
  printf(">>> geometry.exception() { 0x%X at launch index (%d,%d) }\n",
         rtGetExceptionCode(), launch_index.x, launch_index.y);
}


rtBuffer<float3> vertex_buffer;
rtBuffer<float3> normal_buffer;
// rtBuffer<float2> texcoord_buffer;
rtBuffer<int3> index_buffer;

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
// rtDeclareVariable(float3, shading_normal,   attribute shading_normal, );

RT_PROGRAM void triangle_attributes() {
  // printf("triangle_attributes()\n");
// #if OPTIX_VERSION / 10000 >= 6
  const int3 v_idx = index_buffer[rtGetPrimitiveIndex()];
  const float3 v0 = vertex_buffer[v_idx.x];
  const float3 v1 = vertex_buffer[v_idx.y];
  const float3 v2 = vertex_buffer[v_idx.z];
  const float3 Ng = optix::cross(v1 - v0, v2 - v0);

  geometric_normal = optix::normalize(Ng);

  // barycentrics = rtGetTriangleBarycentrics();
  // // texcoord = make_float3(barycentrics.x, barycentrics.y, 0.0f);
  //
  // if (normal_buffer.size() == 0) {
  //   shading_normal = geometric_normal;
  // } else {
  //   shading_normal = normal_buffer[v_idx.y] * barycentrics.x + normal_buffer[v_idx.z] * barycentrics.y
  //                    + normal_buffer[v_idx.x] * (1.0f - barycentrics.x - barycentrics.y);
  // }

  // if (texcoord_buffer.size() == 0) {
  //   texcoord = make_float3(0.0f, 0.0f, 0.0f);
  // } else {
  //   const float2 t0 = texcoord_buffer[v_idx.x];
  //   const float2 t1 = texcoord_buffer[v_idx.y];
  //   const float2 t2 = texcoord_buffer[v_idx.z];
  //   texcoord = make_float3(t1 * barycentrics.x + t2 * barycentrics.y + t0 * (1.0f - barycentrics.x - barycentrics.y));
  // }
// #endif
}
