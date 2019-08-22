#pragma once

#include <optix_world.h>
#include <optix_device.h>

#include "openmc/surface.h"

using namespace openmc;

template<int i> float
__device__ __forceinline__
axis_aligned_plane_distance(Position_& r, Direction_& u, bool coincident, float offset)
{
  const float f = offset - r[i];
  if (coincident or std::abs(f) < FP_COINCIDENT or u[i] == 0.0) return INFTY;
  const float d = f / u[i];
  if (d < 0.0) return INFTY;
  return d;
}

__device__ __forceinline__
float _distance_xplane(Surface_& s, Position_& r, Direction_& u, bool coincident)
{
  return axis_aligned_plane_distance<0>(r, u, coincident, s.xplane_x0_);
}

__device__ __forceinline__
float _evaluate_xplane(Surface_& s, Position_& r)
{
  return r.x - s.xplane_x0_;
}

__device__ __forceinline__
Direction_ _normal_xplane(Surface_& s, Position_& r)
{
  return {1., 0., 0.};
}

__device__ __forceinline__
float _distance_yplane(Surface_& s, Position_& r, Direction_& u, bool coincident)
{
  return axis_aligned_plane_distance<1>(r, u, coincident, s.yplane_y0_);
}

__device__ __forceinline__
float _evaluate_yplane(Surface_& s, Position_& r)
{
  return r.y - s.yplane_y0_;
}

__device__ __forceinline__
Direction_ _normal_yplane(Surface_& s, Position_& r)
{
  return {0., 1., 0.};
}

__device__ __forceinline__
float _distance_zplane(Surface_& s, Position_& r, Direction_& u, bool coincident)
{
  return axis_aligned_plane_distance<2>(r, u, coincident, s.zplane_z0_);
}

__device__ __forceinline__
float _evaluate_zplane(Surface_& s, Position_& r)
{
  return r.z - s.zplane_z0_;
}

__device__ __forceinline__
Direction_ _normal_zplane(Surface_& s, Position_& r)
{
  return {0., 0., 1.};
}

__device__ __forceinline__
float _distance_sphere(Surface_& s, Position_& r, Direction_& u, bool coincident)
{
  const float x = r.x - s.sphere_x0_;
  const float y = r.y - s.sphere_y0_;
  const float z = r.z - s.sphere_z0_;
  const float k = x*u.x + y*u.y + z*u.z;
  const float c = x*x + y*y + z*z - s.sphere_radius_*s.sphere_radius_;
  const float quad = k*k - c;

  if (quad < 0.0f) {
    // No intersection with sphere.
    return (float) INFTY;

  } else if (coincident or fabsf(c) < (float) FP_COINCIDENT) {
    // Particle is on the sphere, thus one distance is positive/negative and
    // the other is zero. The sign of k determines if we are facing in or out.
    if (k >= 0.0f) {
      return (float) INFTY;
    } else {
      return -k + sqrtf(quad);
    }

  } else if (c < 0.0f) {
    // Particle is inside the sphere, thus one distance must be negative and
    // one must be positive. The positive distance will be the one with
    // negative sign on sqrt(quad)
    return -k + sqrtf(quad);

  } else {
    // Particle is outside the sphere, thus both distances are either positive
    // or negative. If positive, the smaller distance is the one with positive
    // sign on sqrt(quad).
    const float d = -k - sqrtf(quad);
    if (d < 0.0f) return (float) INFTY;
    return d;
  }
}

__device__ __forceinline__
float _evaluate_sphere(Surface_& s, Position_& r)
{
  const float x = r.x - s.sphere_x0_;
  const float y = r.y - s.sphere_y0_;
  const float z = r.z - s.sphere_z0_;
  return x*x + y*y + z*z - s.sphere_radius_*s.sphere_radius_;
}

__device__ __forceinline__
Direction_ _normal_sphere(Surface_& s, Position_& r)
{
  return {2.0f*(r.x - s.sphere_x0_), 2.0f*(r.y - s.sphere_y0_), 2.0f*(r.z - s.sphere_z0_)};
}

__device__ __forceinline__
float _distance(Surface_& s, Position_& r, Direction_& u, bool coincident) {
  switch (s.type) {
    case Surface_::Type::xplane:
      return _distance_xplane(s, r, u, coincident);
    case Surface_::Type::yplane:
      return _distance_yplane(s, r, u, coincident);
    case Surface_::Type::zplane:
      return _distance_zplane(s, r, u, coincident);
    case Surface_::Type::sphere:
      return _distance_sphere(s, r, u, coincident);
    default:
      printf("ERROR: Unsupported surface type\n");
  }
}

__device__ __forceinline__
float _evaluate(Surface_& s, Position_ r) {
  switch (s.type) {
    case Surface_::Type::xplane:
      return _evaluate_xplane(s, r);
    case Surface_::Type::yplane:
      return _evaluate_yplane(s, r);
    case Surface_::Type::zplane:
      return _evaluate_zplane(s, r);
    case Surface_::Type::sphere:
      return _evaluate_sphere(s, r);
    default:
      printf("ERROR: Unsupported surface type\n");
  }
}

__device__ __forceinline__
Direction_ _normal(Surface_& s, Position_ r) {
  switch (s.type) {
    case Surface_::Type::xplane:
      return _normal_xplane(s, r);
    case Surface_::Type::yplane:
      return _normal_yplane(s, r);
    case Surface_::Type::zplane:
      return _normal_zplane(s, r);
    case Surface_::Type::sphere:
      return _normal_sphere(s, r);
    default:
      printf("ERROR: Unsupported surface type\n");
  }
}

__device__ __forceinline__
bool _sense(Surface_& s, Position_& r, Direction_& u)
{
  // Evaluate the surface equation at the particle's coordinates to determine
  // which side the particle is on.
  const float f = _evaluate(s, r);

  // Check which side of surface the point is on.
  if (fabsf(f) < (float) FP_COINCIDENT) {
    // Particle may be coincident with this surface. To determine the sense, we
    // look at the direction of the particle relative to the surface normal (by
    // default in the positive direction) via their dot product.
    return u.dot(_normal(s, r)) > 0.0f;
  }
  return f > 0.0f;
}