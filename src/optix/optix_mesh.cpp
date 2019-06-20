#include "openmc/optix/optix_mesh.h"
#include "openmc/optix/optix_cell.h"
// #include "openmc/optix/utils.h"

namespace openmc {

using namespace optix;

Mesh::Mesh(const std::string &filename) {
  // std::string err, warn;
  //
  // bool ret = tinyobj::LoadObj(
  //   shapes,
  //   materials,
  //   err,
  //   filename.c_str(),
  //   directoryOfFilePath2( filename ).c_str()
  // );
  //
  // if (!err.empty())
  //   std::cerr << err << std::endl;
  //
  // if (!ret)
  //   throw std::runtime_error("Mesh: " + err);
}

}