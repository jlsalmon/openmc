#pragma once

#include <optix_world.h>

#include "distribution_angle.cu"
#include "distribution_energy.cu"
#include "random_lcg.cu"

#include "openmc/secondary_kalbach.h"


__device__ __forceinline__
void _sample_kalbach_mann(const KalbachMann_& km, double E_in, double& E_out, double& mu) {
  // TODO
  printf("KALBACH MANN\n");
}
