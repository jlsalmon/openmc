#pragma once

#include <optix_world.h>
#include <optix_cuda.h>

#include "variables.cu"
#include "random_lcg.cu"

__forceinline__ __device__
float maxwell_spectrum(float T) {
  // Set the random numbers
  float r1 = prn();
  float r2 = prn();
  float r3 = prn();

  // determine cosine of pi/2*r
  float c = cosf(M_PIf / 2.f * r3);

  // Determine outgoing energy
  float E_out = -T * (log(r1) + log(r2) * c * c);

  return E_out;
}

__forceinline__ __device__
float watt_spectrum(float a, float b) {
  float w = maxwell_spectrum(a);
  float E_out = w + 0.25f * a * a * b + (2.f * prn() - 1.f) * sqrt(a * a * b * w);

  return E_out;
}

__forceinline__ __device__
Direction_ rotate_angle(Direction_ u, float mu, const float* phi)
{
  // Sample azimuthal angle in [0,2pi) if none provided
  float phi_;
  if (phi != nullptr) {
    phi_ = (*phi);
  } else {
    phi_ = 2.0f*M_PIf*prn();
  }

  // Precompute factors to save flops
  float sinphi = sinf(phi_);
  float cosphi = cosf(phi_);
  float a = sqrtf(fmaxf(0.f, 1.f - mu*mu));
  float b = sqrtf(fmaxf(0.f, 1.f - u.z*u.z));

  // Need to treat special case where sqrt(1 - w**2) is close to zero by
  // expanding about the v component rather than the w component
  if (b > 1e-10f) {
    return {mu*u.x + a*(u.x*u.z*cosphi - u.y*sinphi) / b,
            mu*u.y + a*(u.y*u.z*cosphi + u.x*sinphi) / b,
            mu*u.z - a*b*cosphi};
  } else {
    b = sqrtf(1.f - u.y*u.y);
    return {mu*u.x + a*(u.x*u.y*cosphi + u.z*sinphi) / b,
            mu*u.y - a*b*cosphi,
            mu*u.z + a*(u.y*u.z*cosphi - u.x*sinphi) / b};
  }
}

__device__ __forceinline__
int _lower_bound(int lo, int hi, rtBufferId<float, 1> arr, float target) {
  // int l = 0;
  // int h = hi; // Not n - 1
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (target <= arr[mid]) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo - 1;
}