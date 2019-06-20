#ifndef OPENMC_OPTIX_GEOMETRY_H
#define OPENMC_OPTIX_GEOMETRY_H

#include <optix_world.h>
#include <sutil.h>
#include <Mesh.h>
#include "openmc/optix/optix_mesh.h"

namespace openmc {

using namespace optix;

class OptiXGeometry: public MeshVisitor {
private:
  Context context;
  // Mesh *mesh;

  Buffer instance_id_buffer;
  Buffer intersection_distance_buffer;
  Buffer num_hits_buffer;

  const char *const SAMPLE_NAME = "cuda_compile_ptx_1";
  const char *ptx = sutil::getPtxString(SAMPLE_NAME, "mesh.cu");

  int universe_id = 0;

  void create_context();
  // void load_mesh(const std::string &filename);

  void visit(int index, tinyobj::shape_t shape) override;

  void visit(int index, float3 normal) override;

public:
  void load(const std::string& filename);
};

}

#endif //OPENMC_OPTIX_GEOMETRY_H
