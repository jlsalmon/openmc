#pragma once

#include <optix_world.h>

#include "variables.cu"

// Constants
__device__ const int _N_STREAMS         {6};
__device__ const int _STREAM_TRACKING   {0};
__device__ const int _STREAM_TALLIES    {1};
__device__ const int _STREAM_SOURCE     {2};
__device__ const int _STREAM_URR_PTABLE {3};
__device__ const int _STREAM_VOLUME     {4};
__device__ const int _STREAM_PHOTON     {5};

// Starting seed
__device__ const int64_t  seed {1};

// LCG parameters
__device__ const uint64_t prn_mult   {2806196910506780709LL};   // multiplication
//   factor, g
__device__ const uint64_t prn_add    {1};                       // additive factor, c
__device__ const uint64_t prn_mod    {0x8000000000000000};      // 2^63
__device__ const uint64_t prn_mask   {0x7fffffffffffffff};      // 2^63 - 1
__device__ const uint64_t prn_stride {152917LL};                // stride between
//   particles
__device__ const double   prn_norm   {1.0 / prn_mod};           // 2^-63

// Current PRNG state
// __device__ uint64_t prn_seed[_N_STREAMS];  // current seed
// __device__ int      stream;               // current RNG stream
// #pragma omp threadprivate(prn_seed, stream)

__forceinline__ __device__
double prn()
{
  // This algorithm uses bit-masking to find the next integer(8) value to be
  // used to calculate the random number.
  int offset = launch_index * _N_STREAMS;
  prn_seed_buffer[offset + stream_buffer[launch_index]] =
    (prn_mult*prn_seed_buffer[offset + stream_buffer[launch_index]] + prn_add) & prn_mask;

  // Once the integer is calculated, we just need to divide by 2**m,
  // represented here as multiplying by a pre-calculated factor
  double result = prn_seed_buffer[offset + stream_buffer[launch_index]] * prn_norm;
  // printf("prn(): %f\n", result);
  return result;
}

__forceinline__ __device__ uint64_t
future_seed(uint64_t n, uint64_t seed)
{
  // Make sure nskip is less than 2^M.
  n &= prn_mask;

  // The algorithm here to determine the parameters used to skip ahead is
  // described in F. Brown, "Random Number Generation with Arbitrary Stride,"
  // Trans. Am. Nucl. Soc. (Nov. 1994). This algorithm is able to skip ahead in
  // O(log2(N)) operations instead of O(N). Basically, it computes parameters G
  // and C which can then be used to find x_N = G*x_0 + C mod 2^M.

  // Initialize constants
  uint64_t g     {prn_mult};
  uint64_t c     {prn_add};
  uint64_t g_new {1};
  uint64_t c_new {0};

  while (n > 0) {
    // Check if the least significant bit is 1.
    if (n & 1) {
      g_new *= g;
      c_new = c_new * g + c;
    }
    c *= (g + 1);
    g *= g;

    // Move bits right, dropping least significant bit.
    n >>= 1;
  }

  // With G and C, we can now find the new seed.
  return (g_new * seed + c_new) & prn_mask;
}

__forceinline__ __device__
double future_prn(int64_t n)
{
  int offset = launch_index * _N_STREAMS;
  return future_seed(static_cast<uint64_t>(n), prn_seed_buffer[offset + stream_buffer[launch_index]]) * prn_norm;
}

__forceinline__ __device__ void
set_particle_seed(int64_t id)
{
  int offset = launch_index * _N_STREAMS;
  for (int i = 0; i < _N_STREAMS; i++) {
    prn_seed_buffer[offset + i] = future_seed(static_cast<uint64_t>(id) * prn_stride, seed + i);
  }
}

__forceinline__ __device__ void
advance_prn_seed(int64_t n)
{
  int offset = launch_index * _N_STREAMS;
  prn_seed_buffer[offset + stream_buffer[launch_index]] =
    future_seed(static_cast<uint64_t>(n), prn_seed_buffer[offset + stream_buffer[launch_index]]);
}

__forceinline__ __device__ void
prn_set_stream(int i)
{
  stream_buffer[launch_index] = i;
}
