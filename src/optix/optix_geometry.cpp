#include "openmc/material.h"
#include "openmc/geometry.h"
#include "openmc/geometry_aux.h"
#include "openmc/error.h"
#include "openmc/settings.h"

#include "openmc/optix/optix_geometry.h"
#include "openmc/optix/optix_cell.h"
#include "openmc/optix/optix_surface.h"

#include <optix_world.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_vector_types.h>
#include <cuda_runtime_api.h>
#include <OptiXMesh.h>
#include "cuda/common.h"
#include "cuda/helpers.h"
#include "sutil.h"

namespace openmc {

using namespace optix;

void OptiXGeometry::load(const std::string &filename) {
  write_message("Loading OptiX geometry...", 5);

  create_context();
  load_mesh(filename);

  printf(">>> num cells: %ld, num surfaces: %ld\n",
         model::cells.size(), model::surfaces.size());
}

void OptiXGeometry::create_context() {
  write_message("Creating OptiX context...", 5);

  int rtx = settings::rtx ? 1 : 0;
  if (rtGlobalSetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(rtx), &rtx) != RT_SUCCESS) {
    printf("Error setting RT_GLOBAL_ATTRIBUTE_ENABLE_RTX!\n");
  } else {
    printf("RTX execution mode is %s\n", (rtx) ? "ON" : "OFF");
  }

  context = Context::create();
  context->setEntryPointCount(3);
  context->setRayTypeCount(1);
  // context->setStackSize(2800);
  // context->setMaxTraceDepth(12);

  size_t free, total;
  cudaMemGetInfo(&free, &total);
  printf("GPU memory: %zu of %zu bytes free\n", free, total);

  // Figure out how much heap memory will be needed
  // cudaDeviceSetLimit(cudaLimit::cudaLimitMallocHeapSize, free);

  UsageReportLogger *logger;
  int usage_report_level = 2;
  // context->setUsageReportCallback(usage_report_callback, usage_report_level, logger);

  // FIXME: enabling exceptions somehow hides a misaligned address error...
  // context->setExceptionEnabled(RT_EXCEPTION_ALL, true);
  // context->setPrintEnabled(true);
  // context->setPrintLaunchIndex(29783, 0, 0);

  context["scene_epsilon"]->setFloat(1.e-4f);

  // Single-particle tracking
  context->setRayGenerationProgram(0, context->createProgramFromPTXString(ptx_basic, "generate_ray_basic"));
  context->setExceptionProgram(0, context->createProgramFromPTXString(ptx_basic, "exception_basic"));
  context->setMissProgram(0, context->createProgramFromPTXString(ptx_basic, "miss_basic"));

  // Source sampling
  context->setRayGenerationProgram(1, context->createProgramFromPTXString(ptx_source, "sample_source"));
  context->setExceptionProgram(1, context->createProgramFromPTXString(ptx_source, "exception"));
  // context->setMissProgram(1, context->createProgramFromPTXString(ptx_sample_source, "miss_sample"));

  // Particle transport
  context->setRayGenerationProgram(2, context->createProgramFromPTXString(ptx_simulation, "simulate_particle"));
  context->setExceptionProgram(2, context->createProgramFromPTXString(ptx_simulation, "exception"));
  // context->setMissProgram(2, context->createProgramFromPTXString(ptx_simulation, "miss"));

  Program ray_gen_basic = context->getRayGenerationProgram(0);

  instance_id_buffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_INT, 1);
  ray_gen_basic["instance_id_buffer"]->set(instance_id_buffer);
  intersection_distance_buffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, 1);
  ray_gen_basic["intersection_distance_buffer"]->set(intersection_distance_buffer);
  num_hits_buffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_INT, 1);
  ray_gen_basic["num_hits_buffer"]->set(num_hits_buffer);
}

void OptiXGeometry::load_mesh(const std::string &filename) {
  mesh.context = context;
  mesh.visitor = this;
  mesh.use_tri_api = settings::use_tri_api;
  mesh.ignore_mats = false;

  // TODO: the sample_source and transport entry points will need their own
  // ch/ah programs (and hence materials)
  // bool use_gpu = false;
  // if (use_gpu) {
  //   mesh.closest_hit = context->createProgramFromPTXString(ptx_sample_source, "closest_hit");
  //   mesh.any_hit = context->createProgramFromPTXString(ptx_sample_source, "any_hit");
  // } else {
  //   mesh.closest_hit = context->createProgramFromPTXString(ptx_basic, "closest_hit");
  //   mesh.any_hit = context->createProgramFromPTXString(ptx_basic, "any_hit");
  // }

  // optix::Material basic_mat = context->createMaterial();
  // basic_mat->setClosestHitProgram(0u, context->createProgramFromPTXString(ptx_basic, "closest_hit_basic"));
  // basic_mat->setAnyHitProgram(0u, context->createProgramFromPTXString(ptx_basic, "any_hit_basic"));
  //
  // optix::Material sample_source_mat = context->createMaterial();
  // sample_source_mat->setClosestHitProgram(1u, context->createProgramFromPTXString(ptx_sample_source, "closest_hit_sample"));
  // sample_source_mat->setAnyHitProgram(1u, context->createProgramFromPTXString(ptx_sample_source, "any_hit_sample") );
  //
  // std::vector<optix::Material> materials;
  // materials.push_back(basic_mat);
  // materials.push_back(sample_source_mat);
  // mesh.materials = materials;

  mesh.closest_hit = context->createProgramFromPTXString(ptx_basic, "closest_hit_basic");
  mesh.any_hit = context->createProgramFromPTXString(ptx_basic, "any_hit_basic");

  loadMesh(filename, mesh);

  GeometryGroup geometry_group = context->createGeometryGroup();
  geometry_group->addChild(mesh.geom_instance);
  geometry_group->setAcceleration(context->createAcceleration("Trbvh"));
  context["top_object"]->set(geometry_group);

  context["normal_buffer"]->setBuffer(mesh.geom_instance["normal_buffer"]->getBuffer());

  // context->validate();
}

void OptiXGeometry::visit(int shape_id, tinyobj::shape_t shape) {
  Cell *cell;

  bool use_gpu = false;

  if (use_gpu) {
    cell = new DummyCell();
  } else {
    cell = new OptiXCell();
    ((OptiXCell *) cell)->context = context;
    ((OptiXCell *) cell)->instance_id_buffer = instance_id_buffer;
    ((OptiXCell *) cell)->intersection_distance_buffer = intersection_distance_buffer;
    ((OptiXCell *) cell)->num_hits_buffer = num_hits_buffer;
  }

  cell->id_ = shape_id + 1;
  cell->universe_ = universe_id;
  cell->fill_ = C_NONE;
  model::cells.emplace_back(cell);
  model::cell_map[cell->id_] = shape_id;

  // Populate the Universe vector and dict
  auto it = model::universe_map.find(universe_id);
  if (it == model::universe_map.end()) {
    model::universes.push_back(std::make_unique<Universe>());
    model::universes.back()->id_ = universe_id;
    model::universes.back()->cells_.push_back(shape_id);
    model::universe_map[universe_id] = model::universes.size() - 1;
  } else {
    model::universes[it->second]->cells_.push_back(shape_id);
  }

  // Materials
  // FIXME: support multiple cells
  if (cell->id_ == 1) { // bounding cube
    cell->material_.push_back(MATERIAL_VOID);
  }
  if (cell->id_ == 2) { // inner object
    cell->material_.push_back(model::materials[0]->id_);
  }

  printf(">>> cell id=%d, shape name: %s\n", cell->id_, shape.name.c_str());
}

void OptiXGeometry::visit(int shape_id, int surface_id, float3 normal) {
  auto *surface = new OptiXSurface();

  surface->id_ = surface_id;
  surface->normal_ = normal;

  // Boundary condition
  if (shape_id == 1) {
    surface->bc_ = BC_VACUUM;
  }

  model::surfaces.emplace_back(surface);
  model::surface_map[surface->id_] = surface_id;

  // printf(">>> surface id=%d\n", surface->id_);
}

}