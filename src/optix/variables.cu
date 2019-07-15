#pragma once

#include <optix_world.h>
#include <optix_cuda.h>

#include "cuda/random.h"

#include "openmc/particle.h"
#include "openmc/cell.h"
#include "openmc/material.h"
#include "openmc/nuclide.h"
#include "openmc/thermal.h"

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
rtDeclareVariable(int, total_gen, ,);
rtDeclareVariable(int, overall_generation, ,);
rtDeclareVariable(unsigned int, num_nuclides, ,);
rtDeclareVariable(int, temperature_method, ,);
rtDeclareVariable(float, log_spacing, ,);
rtDeclareVariable(float, energy_min_neutron, ,);
rtDeclareVariable(float, energy_max_neutron, ,);
rtDeclareVariable(float, keff, ,);
rtDeclareVariable(Material_, material, ,);
rtDeclareVariable(Material::ThermalTable, thermal_table, ,);

// rtBuffer<rtBufferId<int,1>, 1> input_buffers;

// Nuclide
rtDeclareVariable(Nuclide_, nuclide, ,);
// rtBuffer<Reaction_> reactions_buffer;

// Input buffers
rtBuffer<uint64_t> prn_seed_buffer;
rtBuffer<int> stream_buffer;
rtBuffer<float3> normal_buffer;
rtBuffer<Cell_> cell_buffer;
// rtBuffer<ThermalScattering> thermal_scatt_buffer;
// rtBuffer<double> energy_buffer;
// rtBuffer<int> grid_index_buffer;
// rtBuffer<double> xs_buffer;
// rtBuffer<double> angle_distribution_energy_buffer;
// rtBuffer<int> index_inelastic_scatter_buffer;

// Output buffers
rtBuffer<Particle::Bank> source_bank_buffer;
rtBuffer<Particle::Bank> fission_bank_buffer;
rtBuffer<Particle::Bank> secondary_bank_buffer;
