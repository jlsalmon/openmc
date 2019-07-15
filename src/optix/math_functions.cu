#pragma once

#include <optix_world.h>
#include <optix_cuda.h>

#include "variables.cu"
#include "random_lcg.cu"

__forceinline__ __device__
double maxwell_spectrum(double T) {
  // Set the random numbers
  double r1 = prn();
  double r2 = prn();
  double r3 = prn();

  // determine cosine of pi/2*r
  double c = cosf(PI / 2. * r3);

  // Determine outgoing energy
  double E_out = -T * (log(r1) + log(r2) * c * c);

  return E_out;
}

__forceinline__ __device__
double watt_spectrum(double a, double b) {
  double w = maxwell_spectrum(a);
  double E_out = w + 0.25 * a * a * b + (2. * prn() - 1.) * sqrt(a * a * b * w);

  return E_out;
}

__forceinline__ __device__
Direction rotate_angle(Direction u, double mu, const double* phi)
{
  // Sample azimuthal angle in [0,2pi) if none provided
  double phi_;
  if (phi != nullptr) {
    phi_ = (*phi);
  } else {
    phi_ = 2.0*PI*prn();
  }

  // Precompute factors to save flops
  double sinphi = sinf(phi_);
  double cosphi = cosf(phi_);
  double a = sqrt(fmax(0., 1. - mu*mu));
  double b = sqrt(fmax(0., 1. - u.z*u.z));

  // Need to treat special case where sqrt(1 - w**2) is close to zero by
  // expanding about the v component rather than the w component
  if (b > 1e-10) {
    return {mu*u.x + a*(u.x*u.z*cosphi - u.y*sinphi) / b,
            mu*u.y + a*(u.y*u.z*cosphi + u.x*sinphi) / b,
            mu*u.z - a*b*cosphi};
  } else {
    b = sqrt(1. - u.y*u.y);
    return {mu*u.x + a*(u.x*u.y*cosphi + u.z*sinphi) / b,
            mu*u.y - a*b*cosphi,
            mu*u.z + a*(u.y*u.z*cosphi - u.x*sinphi) / b};
  }
}

__device__ __forceinline__
int _lower_bound(int lo, int hi, rtBufferId<double, 1> arr, double target) {
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