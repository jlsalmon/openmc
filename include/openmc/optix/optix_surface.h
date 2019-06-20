#ifndef OPENMC_OPTIX_SURFACE_H
#define OPENMC_OPTIX_SURFACE_H

#include "../surface.h"
#include "tetrahedron.h"

namespace openmc {

class OptiXSurface : public Surface {
public:
  // Tetrahedron *tet;
  float3 normal_;

  OptiXSurface();

  double evaluate(Position r) const;

  double distance(Position r, Direction u, bool coincident) const;

  Direction normal(Position r) const;

  //! Get the bounding box of this surface.
  BoundingBox bounding_box() const;

  void to_hdf5(hid_t group_id) const;
};
}

#endif //OPENMC_OPTIX_SURFACE_H
