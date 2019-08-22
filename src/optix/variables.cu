#pragma once

#include <optix_world.h>
#include <optix_cuda.h>

#include "cuda/random.h"

#include "openmc/simulation.h"
#include "openmc/particle.h"
#include "openmc/cell.h"
#include "openmc/material.h"
#include "openmc/nuclide.h"
#include "openmc/source.h"

#define _BC_TRANSMIT 0
#define _BC_VACUUM 1
#define _BC_REFLECT 2
#define _BC_PERIODIC 3

#define OPENMC_E_GEOMETRY_ -8

using namespace optix;
using namespace openmc;

struct PerRayData {
  int surface_id;
  float intersection_distance;
  float3 position;
  bool hit;
};

// OptiX semantic variables
rtDeclareVariable(Ray, ray, rtCurrentRay,);
rtDeclareVariable(PerRayData, payload, rtPayload,);
rtDeclareVariable(unsigned int, launch_index, rtLaunchIndex,);
rtDeclareVariable(float, intersection_distance, rtIntersectionDistance,);
rtDeclareVariable(rtObject, top_object, ,);
rtDeclareVariable(float, scene_epsilon, ,);

// User variables
rtDeclareVariable(unsigned int, n_particles, ,);
// rtDeclareVariable(int, total_gen, ,);
// rtDeclareVariable(int, overall_generation, ,);
rtDeclareVariable(unsigned int, num_nuclides, ,);
rtDeclareVariable(int, temperature_method, ,);
rtDeclareVariable(float, log_spacing, ,);
rtDeclareVariable(float, energy_min_neutron, ,);
rtDeclareVariable(float, energy_max_neutron, ,);
rtDeclareVariable(int, use_csg, ,);
// rtDeclareVariable(Material::ThermalTable, thermal_table, ,);

// rtBuffer<rtBufferId<int,1>, 1> input_buffers;

// Nuclide
rtDeclareVariable(Nuclide_, nuclide, ,);
// rtBuffer<Reaction_> reactions_buffer;

// Input buffers
rtBuffer<uint64_t> prn_seed_buffer;
rtBuffer<int> stream_buffer;
rtBuffer<float3> normal_buffer;
rtBuffer<Cell_> cell_buffer;
rtBuffer<Surface_> surface_buffer;
rtBuffer<Material_> material_buffer;
rtBuffer<Particle_> particle_buffer;
rtBuffer<SourceDistribution_> external_sources_buffer;

rtBuffer<simulation_> _simulation;

// Output buffers
rtBuffer<Particle_::Bank_> source_bank_buffer;
rtBuffer<Particle_::Bank_> fission_bank_buffer;
rtBuffer<Particle_::Bank_> secondary_bank_buffer;
rtBuffer<float> global_tally_absorption_buffer;
rtBuffer<float> global_tally_collision_buffer;
rtBuffer<float> global_tally_tracklength_buffer;
rtBuffer<float> global_tally_leakage_buffer;
rtBuffer<float> total_weight_buffer;
