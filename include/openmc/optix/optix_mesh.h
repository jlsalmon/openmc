#ifndef OPENMC_OPTIX_MESH_H
#define OPENMC_OPTIX_MESH_H

#include <tinyobjloader/tiny_obj_loader.h>
#include <iostream>
#include <optix_world.h>

namespace openmc {

using namespace optix;

class Mesh {
public:
  Mesh(const std::string &filename);

  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;

  float*              positions;      // Triangle vertex positions (len num_vertices)
  int32_t*            tri_indices;    // Indices into positions, normals, texcoords
};

}

#endif //OPENMC_OPTIX_MESH_H
