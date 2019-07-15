#ifndef OPENMC_OPTIX_GEOMETRY_H
#define OPENMC_OPTIX_GEOMETRY_H

#include <iomanip>

#include <optix_world.h>
#include <sutil.h>
#include <Mesh.h>
#include <OptiXMesh.h>
#include "openmc/optix/optix_mesh.h"

namespace openmc {

using namespace optix;

class OptiXGeometry;
inline OptiXGeometry *geometry;

struct UsageReportLogger {
  void log(int lvl, const char *tag, const char *msg) {
    std::cout << "[" << lvl << "][" << std::left << std::setw(12) << tag << "] " << msg;
  }
};

inline void usage_report_callback(int lvl, const char *tag, const char *msg, void *cbdata) {
  auto *logger = reinterpret_cast<UsageReportLogger *>( cbdata );
  logger->log(lvl, tag, msg);
}

class OptiXGeometry: public MeshVisitor {
private:
  Buffer instance_id_buffer;
  Buffer intersection_distance_buffer;
  Buffer num_hits_buffer;

  const char *const SAMPLE_NAME = "cuda_compile_ptx_1";
  const char *ptx_basic = sutil::getPtxString(SAMPLE_NAME, "basic.cu");
  const char *ptx_source = sutil::getPtxString(SAMPLE_NAME, "source.cu");
  const char *ptx_simulation = sutil::getPtxString(SAMPLE_NAME, "simulation.cu");

  int universe_id = 0;

  void create_context();
  void load_mesh(const std::string &filename);
  void visit(int shape_id, tinyobj::shape_t shape) override;
  void visit(int shape_id, int surface_id, float3 normal) override;

public:
  OptiXMesh mesh;
  Context context;

  void load(const std::string& filename);
};

}

#endif //OPENMC_OPTIX_GEOMETRY_H
