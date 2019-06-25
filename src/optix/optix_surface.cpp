#include "openmc/error.h"
#include "openmc/optix/optix_surface.h"

namespace openmc {

OptiXSurface::OptiXSurface() : Surface{} {} // empty constructor

double OptiXSurface::evaluate(Position r) const {
  return 0.0;
}

double OptiXSurface::distance(Position r, Direction u, bool coincident) const {
  // moab::ErrorCode rval;
  // moab::EntityHandle surf = dagmc_ptr_->entity_by_index(2, dag_index_);
  // moab::EntityHandle hit_surf;
  // double dist;
  // double pnt[3] = {r.x, r.y, r.z};
  // double dir[3] = {u.x, u.y, u.z};
  // rval = dagmc_ptr_->ray_fire(surf, pnt, dir, hit_surf, dist, NULL, 0, 0);
  // MB_CHK_ERR_CONT(rval);
  // if (dist < 0.0) dist = INFTY;
  // return dist;

  write_message("$$$ surface.distance()", 5);
  return INFTY;
}

Direction OptiXSurface::normal(Position r) const {
  // moab::ErrorCode rval;
  // Direction u;
  // moab::EntityHandle surf = dagmc_ptr_->entity_by_index(2, dag_index_);
  // double pnt[3] = {r.x, r.y, r.z};
  // double dir[3] = {u.x, u.y, u.z};
  // rval = dagmc_ptr_->get_angle(surf, pnt, dir);
  // MB_CHK_ERR_CONT(rval);
  // return u;

  // 0->0, 1->3, 2->6, 3->9
  // float3 normal = tet->normals[this->id_ * 3];

  printf("$$$ surface.normal(): { %f, %f, %f }\n", normal_.x, normal_.y, normal_.z);
  return {normal_.x, normal_.y, normal_.z};
}

BoundingBox OptiXSurface::bounding_box() const {
  // moab::ErrorCode rval;
  // moab::EntityHandle surf = dagmc_ptr_->entity_by_index(2, dag_index_);
  // double min[3], max[3];
  // rval = dagmc_ptr_->getobb(surf, min, max);
  // MB_CHK_ERR_CONT(rval);
  // return {min[0], max[0], min[1], max[1], min[2], max[2]};

  printf("$$$ surface.bounding_box()\n");

  return {0, 0, 0, 0, 0, 0};
}

void OptiXSurface::to_hdf5(hid_t group_id) const {}
}