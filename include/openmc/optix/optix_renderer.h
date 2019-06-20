#ifndef OPENMC_OPTIX_RENDERER_H
#define OPENMC_OPTIX_RENDERER_H

#include <optix_world.h>

namespace openmc {

using namespace optix;

class OptiXRenderer {
public:
  void render();

private:
  const char *const SAMPLE_NAME = "cuda_compile_ptx_1";

  Context context;
  uint32_t width = 768u;
  uint32_t height = 768u;
  float3 camera_up;
  float3 camera_lookat;
  float3 camera_eye;
  Matrix4x4 camera_rotate;
  optix::Material material;
  GeometryInstance tri_gi;

  void createContext();
  void createMaterials();
  GeometryGroup createGeometryTriangles();
  void setupScene();
  void setupCamera();
  void updateCamera();
};

}

#endif //OPENMC_OPTIX_RENDERER_H
