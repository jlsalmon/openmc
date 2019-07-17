#include <optix_world.h>
#include <optix_cuda.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__
#endif

#include "variables.cu"
#include "particle.cu"
#include "material.cu"

#include "openmc/geometry.h"

using namespace optix;
using namespace openmc;

__device__ __forceinline__
void _initialize_history(Particle_& p, int64_t index_source)
{
  // set defaults
  // p->from_source(&simulation::source_bank[index_source - 1]);
  // _from_source(p, &source_bank_buffer[index_source]);

  // set identifier for particle
  // p->id_ = simulation::work_index[mpi::rank] + index_source;
  // p.id_ = index_source;

  // set random number seed
  // int64_t particle_seed = (total_gen + overall_generation - 1) // FIXME: random
  //                         * n_particles + p.id_;
  int64_t particle_seed = index_source;
  set_particle_seed(particle_seed);

  // set particle trace
  // simulation::trace = false; // FIXME
  // if (simulation::current_batch == settings::trace_batch &&
  //     simulation::current_gen == settings::trace_gen &&
  //     p->id_ == settings::trace_particle) simulation::trace = true;

  // Set particle track.
  // p->write_track_ = false; // FIXME
  // if (settings::write_all_tracks) {
  //   p->write_track_ = true;
  // } else if (settings::track_identifiers.size() > 0) {
  //   for (const auto& t : settings::track_identifiers) {
  //     if (simulation::current_batch == t[0] &&
  //         simulation::current_gen == t[1] &&
  //         p->id_ == t[2]) {
  //       p->write_track_ = true;
  //       break;
  //     }
  //   }
  // }
}

RT_PROGRAM void simulate_particle() {
  using namespace openmc;

  Particle_& p = particle_buffer[launch_index];
  _initialize_history(p, launch_index);

  // printf("p.alive: %d\n", p.alive_);
  // printf("p.wgt_: %lf\n", p.wgt_);
  // printf("p.r: (%lf,%lf,%lf)\n", p.r().x, p.r().y, p.r().z);
  // printf("p.u: (%lf,%lf,%lf)\n", p.u().x, p.u().y, p.u().z);
  // printf("p.E_:%lf\n", p.E_);

  // transport particle
  _transport(p);
}


RT_PROGRAM void exception() {
  rtPrintExceptionDetails();
  printf(">>> simulation.exception() { 0x%X at launch index (%d) }\n",
         rtGetExceptionCode(), launch_index);
}

