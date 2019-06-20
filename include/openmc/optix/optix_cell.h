#ifndef OPENMC_OPTIX_CELL_H
#define OPENMC_OPTIX_CELL_H

#include <optix_world.h>
#include "../cell.h"
#include "tetrahedron.h"
#include "optix_mesh.h"

namespace openmc {

class OptiXCell : public Cell {
public:
  mutable optix::Context context;
  mutable optix::Buffer instance_id_buffer;
  mutable optix::Buffer intersection_distance_buffer;
  mutable optix::Buffer num_hits_buffer;

  // mutable Mesh *mesh;

  OptiXCell();

  bool contains(Position r, Direction u, int32_t on_surface) const;

  std::pair<double, int32_t>
  distance(Position r, Direction u, int32_t on_surface) const;

  void to_hdf5(hid_t group_id) const;
};

}

#endif //OPENMC_OPTIX_CELL_H
