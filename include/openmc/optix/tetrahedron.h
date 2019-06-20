#ifndef OPENMC_TETRAHEDRON_H
#define OPENMC_TETRAHEDRON_H

#include <optixu/optixu_vector_types.h>
#include "cuda/helpers.h"

namespace openmc {

using namespace optix;

struct Tetrahedron {
    float3 vertices[12];
    float3 normals[12];
    float2 texcoords[12];
    unsigned indices[12];

    Tetrahedron(const float H, const float3 trans) {
      const float a = (3.0f * H) / sqrtf(6.0f); // Side length
      const float d = a * sqrtf(3.0f) / 6.0f;     // Offset for base vertices from apex

      // There are only four vertex positions, but we will duplicate vertices
      // instead of sharing them among faces.
      const float3 v0 = trans + make_float3(0.0f, 0, H - d);
      const float3 v1 = trans + make_float3(a / 2.0f, 0, -d);
      const float3 v2 = trans + make_float3(-a / 2.0f, 0, -d);
      const float3 v3 = trans + make_float3(0.0f, H, 0.0f);

      printf("tet((%f,%f,%f),(%f,%f,%f),(%f,%f,%f),(%f,%f,%f))\n",
        v0.x,v0.y,v0.z,v1.x,v1.y,v1.z,v2.x,v2.y,v2.z,v3.x,v3.y,v3.z);

      // Bottom face
      vertices[0] = v0;
      vertices[1] = v1;
      vertices[2] = v2;

      // Duplicate the face normals across the vertices.
      float3 n = optix::normalize(optix::cross(v2 - v0, v1 - v0));
      normals[0] = n;
      normals[1] = n;
      normals[2] = n;

      texcoords[0] = make_float2(0.5f, 1.0f);
      texcoords[1] = make_float2(1.0f, 0.0f);
      texcoords[2] = make_float2(0.0f, 0.0f);

      // Left face
      vertices[3] = v3;
      vertices[4] = v2;
      vertices[5] = v0;

      n = optix::normalize(optix::cross(v2 - v3, v0 - v3));
      normals[3] = n;
      normals[4] = n;
      normals[5] = n;

      texcoords[3] = make_float2(0.5f, 1.0f);
      texcoords[4] = make_float2(0.0f, 0.0f);
      texcoords[5] = make_float2(1.0f, 0.0f);

      // Right face
      vertices[6] = v3;
      vertices[7] = v0;
      vertices[8] = v1;

      n = optix::normalize(optix::cross(v0 - v3, v1 - v3));
      normals[6] = n;
      normals[7] = n;
      normals[8] = n;

      texcoords[6] = make_float2(0.5f, 1.0f);
      texcoords[7] = make_float2(0.0f, 0.0f);
      texcoords[8] = make_float2(1.0f, 0.0f);

      // Back face
      vertices[9] = v3;
      vertices[10] = v1;
      vertices[11] = v2;

      n = optix::normalize(optix::cross(v1 - v3, v2 - v3));
      normals[9] = n;
      normals[10] = n;
      normals[11] = n;

      texcoords[9] = make_float2(0.5f, 1.0f);
      texcoords[10] = make_float2(0.0f, 0.0f);
      texcoords[11] = make_float2(1.0f, 0.0f);

      for (int i = 0; i < 12; ++i)
        indices[i] = i;
    }
};

}

#endif //OPENMC_TETRAHEDRON_H
