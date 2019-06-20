#include <iomanip>

#include "openmc/material.h"
#include "openmc/geometry.h"
#include "openmc/geometry_aux.h"
#include "openmc/error.h"

#include "openmc/optix/optix_geometry.h"
#include "openmc/optix/optix_cell.h"
#include "openmc/optix/optix_surface.h"

#include <optix_world.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_vector_types.h>
#include <OptiXMesh.h>
#include "cuda/common.h"
#include "cuda/helpers.h"
#include "sutil.h"

namespace openmc {

using namespace optix;

struct UsageReportLogger {
  void log(int lvl, const char *tag, const char *msg) {
    std::cout << "[" << lvl << "][" << std::left << std::setw(12) << tag << "] " << msg;
  }
};

void usageReportCallback(int lvl, const char *tag, const char *msg, void *cbdata) {
  auto *logger = reinterpret_cast<UsageReportLogger *>( cbdata );
  logger->log(lvl, tag, msg);
}

void OptiXGeometry::visit(int index, tinyobj::shape_t shape) {
  auto *cell = new OptiXCell();
  cell->context = context;
  cell->instance_id_buffer = instance_id_buffer;
  cell->intersection_distance_buffer = intersection_distance_buffer;
  cell->num_hits_buffer = num_hits_buffer;
  // cell->mesh = mesh;
  cell->id_ = index + 1;
  cell->universe_ = universe_id;
  cell->fill_ = C_NONE;
  model::cells.emplace_back(cell);
  model::cell_map[cell->id_] = index;

  // Populate the Universe vector and dict
  auto it = model::universe_map.find(universe_id);
  if (it == model::universe_map.end()) {
    model::universes.push_back(std::make_unique<Universe>());
    model::universes.back()->id_ = universe_id;
    model::universes.back()->cells_.push_back(index);
    model::universe_map[universe_id] = model::universes.size() - 1;
  } else {
    model::universes[it->second]->cells_.push_back(index);
  }

  // TODO: Materials
  if (index == 0) {
    cell->material_.push_back(model::materials[0]->id_);
  }

  printf(">>> cell id=%d\n", cell->id_);
}

void OptiXGeometry::visit(int index, float3 normal) {
  auto *surface = new OptiXSurface();

  // TODO: how to ensure that OpenMC surface IDs match OptiX primitive indices?


  surface->id_ = index;
  surface->normal_ = normal;

  // TODO: Boundary condition
  if (index >= 12) {
    surface->bc_ = BC_VACUUM;
  }

  model::surfaces.emplace_back(surface);
  model::surface_map[surface->id_] = index;

  printf(">>> surface id=%d\n", surface->id_);
}


void OptiXGeometry::load(const std::string &filename) {
  write_message("Loading OptiX geometry...", 5);

  create_context();
  OptiXMesh mesh;
  mesh.context = context;
  mesh.visitor = this;
  mesh.use_tri_api = true;
  mesh.ignore_mats = true;
  mesh.closest_hit = context->createProgramFromPTXString(ptx, "closest_hit");
  mesh.any_hit = context->createProgramFromPTXString(ptx, "any_hit");
  loadMesh( filename, mesh );

  GeometryGroup geometry_group = context->createGeometryGroup();
  geometry_group->addChild(mesh.geom_instance);
  geometry_group->setAcceleration(context->createAcceleration("Trbvh"));
  context["top_object"]->set(geometry_group);
  context->validate();


  int num_shapes = 2;
  int num_triangles = mesh.num_triangles;

  // Create OpenMC cells and surfaces

  for (int i = 0; i < num_shapes; i++) {


    for (int j = 0; j < num_triangles; j++) {

    }
  }

  model::root_universe = find_root_universe();
}

void OptiXGeometry::create_context() {
  write_message("Creating OptiX context...", 5);

#if OPTIX_VERSION / 10000 >= 6
  int rtx = 1;
  if (rtGlobalSetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(rtx), &rtx) != RT_SUCCESS) {
    printf("Error setting RT_GLOBAL_ATTRIBUTE_ENABLE_RTX!\n");
  } else {
    printf("RTX execution mode is %s\n", (rtx) ? "ON" : "OFF");
  }
#endif

  context = Context::create();
  context->setRayTypeCount(2);
  context->setEntryPointCount(1);
  context->setStackSize(2800);
#if OPTIX_VERSION / 10000 > 6
  context->setMaxTraceDepth( 12 );
#endif

  UsageReportLogger *logger;
  int usage_report_level = 0;
  context->setUsageReportCallback(usageReportCallback, usage_report_level, logger);

  context["scene_epsilon"]->setFloat(1.e-4f);

  instance_id_buffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_INT, 1);
  context["instance_id_buffer"]->set(instance_id_buffer);

  intersection_distance_buffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, 1);
  context["intersection_distance_buffer"]->set(intersection_distance_buffer);

  num_hits_buffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_INT, 1);
  context["num_hits_buffer"]->set(num_hits_buffer);

  // Programs
  context->setRayGenerationProgram(0, context->createProgramFromPTXString(ptx, "generate_ray"));
  context->setExceptionProgram(0, context->createProgramFromPTXString(ptx, "exception"));
  context->setMissProgram(0, context->createProgramFromPTXString(ptx, "miss"));
}

}