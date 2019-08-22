#ifndef OPENMC_POSITION_H
#define OPENMC_POSITION_H

#include <array>
#include <cmath> // for sqrt
#include <stdexcept> // for out_of_range
#include <vector>

#include <optix_world.h>

namespace openmc {

//==============================================================================
//! Type representing a position in Cartesian coordinates
//==============================================================================

struct Position {
  // Constructors
  __host__ __forceinline__ __device__ Position() = default;
  __host__ __forceinline__ __device__ Position(double x_, double y_, double z_) : x{x_}, y{y_}, z{z_} { };
  __host__ __forceinline__ __device__ Position(const double xyz[]) : x{xyz[0]}, y{xyz[1]}, z{xyz[2]} { };
  Position(const std::vector<double>& xyz) : x{xyz[0]}, y{xyz[1]}, z{xyz[2]} { };
  Position(const std::array<double, 3>& xyz) : x{xyz[0]}, y{xyz[1]}, z{xyz[2]} { };

  // Unary operators
  __host__ __forceinline__ __device__ Position& operator+=(Position other)
  {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
  }
  __host__ __forceinline__ __device__ Position& operator+=(double v)
  {
    x += v;
    y += v;
    z += v;
    return *this;
  }
  __host__ __forceinline__ __device__ Position& operator-=(Position other)
  {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
  }
  __host__ __forceinline__ __device__ Position& operator-=(double v)
  {
    x -= v;
    y -= v;
    z -= v;
    return *this;
  }
  __host__ __forceinline__ __device__ Position& operator*=(Position other)
  {
    x *= other.x;
    y *= other.y;
    z *= other.z;
    return *this;
  }
  __host__ __forceinline__ __device__ Position& operator*=(double v)
  {
    x *= v;
    y *= v;
    z *= v;
    return *this;
  }
  __host__ __forceinline__ __device__ Position& operator/=(Position other)
  {
    x /= other.x;
    y /= other.y;
    z /= other.z;
    return *this;
  }
  __host__ __forceinline__ __device__ Position& operator/=(double v)
  {
    x /= v;
    y /= v;
    z /= v;
    return *this;
  }
  __host__ __forceinline__ __device__ Position operator-() const
  {
    return {-x, -y, -z};
  }

  const double& operator[](int i) const {
    switch (i) {
      case 0: return x;
      case 1: return y;
      case 2: return z;
      default:
        throw std::out_of_range{"Index in Position must be between 0 and 2."};
    }
  }
  double& operator[](int i) {
    switch (i) {
      case 0: return x;
      case 1: return y;
      case 2: return z;
      default:
        throw std::out_of_range{"Index in Position must be between 0 and 2."};
    }
  }

  // Other member functions

  //! Dot product of two vectors
  //! \param[in] other Vector to take dot product with
  //! \result Resulting dot product
  __host__ __forceinline__ __device__ double dot(Position other) {
    return x*other.x + y*other.y + z*other.z;
  }
  __host__ __forceinline__ __device__ double norm() {
    return std::sqrt(x*x + y*y + z*z);
  }

  // Data members
  double x = 0.;
  double y = 0.;
  double z = 0.;
};

// Binary operators
__host__ __forceinline__ __device__ Position operator+(Position a, Position b) { return a += b; }
__host__ __forceinline__ __device__ Position operator+(Position a, double b)   { return a += b; }
__host__ __forceinline__ __device__ Position operator+(double a, Position b)   { return b += a; }

__host__ __forceinline__ __device__ Position operator-(Position a, Position b) { return a -= b; }
__host__ __forceinline__ __device__ Position operator-(Position a, double b)   { return a -= b; }
__host__ __forceinline__ __device__ Position operator-(double a, Position b)   { return b -= a; }

__host__ __forceinline__ __device__ Position operator*(Position a, Position b) { return a *= b; }
__host__ __forceinline__ __device__ Position operator*(Position a, double b)   { return a *= b; }
__host__ __forceinline__ __device__ Position operator*(double a, Position b)   { return b *= a; }

__host__ __forceinline__ __device__ Position operator/(Position a, Position b) { return a /= b; }
__host__ __forceinline__ __device__ Position operator/(Position a, double b)   { return a /= b; }
__host__ __forceinline__ __device__ Position operator/(double a, Position b)   { return b /= a; }

inline bool operator==(Position a, Position b)
{return a.x == b.x && a.y == b.y && a.z == b.z;}

inline bool operator!=(Position a, Position b)
{return a.x != b.x || a.y != b.y || a.z != b.z;}

//==============================================================================
//! Type representing a vector direction in Cartesian coordinates
//==============================================================================

using Direction = Position;

struct Position_ {
  // Constructors
  __host__ __forceinline__ __device__ Position_() = default;
  __host__ __forceinline__ __device__ Position_(float x_, float y_, float z_) : x{x_}, y{y_}, z{z_} { };
  __host__ __forceinline__ __device__ Position_(const float xyz[]) : x{xyz[0]}, y{xyz[1]}, z{xyz[2]} { };

  // Unary operators
  __host__ __forceinline__ __device__ Position_& operator+=(Position_ other)
  {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
  }
  __host__ __forceinline__ __device__ Position_& operator+=(float v)
  {
    x += v;
    y += v;
    z += v;
    return *this;
  }
  __host__ __forceinline__ __device__ Position_& operator-=(Position_ other)
  {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
  }
  __host__ __forceinline__ __device__ Position_& operator-=(float v)
  {
    x -= v;
    y -= v;
    z -= v;
    return *this;
  }
  __host__ __forceinline__ __device__ Position_& operator*=(Position_ other)
  {
    x *= other.x;
    y *= other.y;
    z *= other.z;
    return *this;
  }
  __host__ __forceinline__ __device__ Position_& operator*=(float v)
  {
    x *= v;
    y *= v;
    z *= v;
    return *this;
  }
  __host__ __forceinline__ __device__ Position_& operator/=(Position_ other)
  {
    x /= other.x;
    y /= other.y;
    z /= other.z;
    return *this;
  }
  __host__ __forceinline__ __device__ Position_& operator/=(float v)
  {
    x /= v;
    y /= v;
    z /= v;
    return *this;
  }
  __host__ __forceinline__ __device__ Position_ operator-() const
  {
    return {-x, -y, -z};
  }

  __host__ __forceinline__ __device__
  const float& operator[](int i) const {
    switch (i) {
      case 0: return x;
      case 1: return y;
      case 2: return z;
      default:
        printf("ERROR: Index in Position must be between 0 and 2.\n");
    }
  }

  // Other member functions

  //! Dot product of two vectors
  //! \param[in] other Vector to take dot product with
  //! \result Resulting dot product
  __host__ __forceinline__ __device__ float dot(Position_ other) {
    return x*other.x + y*other.y + z*other.z;
  }
  __host__ __forceinline__ __device__ float norm() {
    return std::sqrt(x*x + y*y + z*z);
  }

  // Data members
  float x = 0.f;
  float y = 0.f;
  float z = 0.f;
};

// Binary operators
__host__ __forceinline__ __device__ Position_ operator+(Position_ a, Position_ b) { return a += b; }
__host__ __forceinline__ __device__ Position_ operator+(Position_ a, float b)   { return a += b; }
__host__ __forceinline__ __device__ Position_ operator+(float a, Position_ b)   { return b += a; }

__host__ __forceinline__ __device__ Position_ operator-(Position_ a, Position_ b) { return a -= b; }
__host__ __forceinline__ __device__ Position_ operator-(Position_ a, float b)   { return a -= b; }
__host__ __forceinline__ __device__ Position_ operator-(float a, Position_ b)   { return b -= a; }

__host__ __forceinline__ __device__ Position_ operator*(Position_ a, Position_ b) { return a *= b; }
__host__ __forceinline__ __device__ Position_ operator*(Position_ a, float b)   { return a *= b; }
__host__ __forceinline__ __device__ Position_ operator*(float a, Position_ b)   { return b *= a; }

__host__ __forceinline__ __device__ Position_ operator/(Position_ a, Position_ b) { return a /= b; }
__host__ __forceinline__ __device__ Position_ operator/(Position_ a, float b)   { return a /= b; }
__host__ __forceinline__ __device__ Position_ operator/(float a, Position_ b)   { return b /= a; }

inline bool operator==(Position_ a, Position_ b)
{return a.x == b.x && a.y == b.y && a.z == b.z;}

inline bool operator!=(Position_ a, Position_ b)
{return a.x != b.x || a.y != b.y || a.z != b.z;}

using Direction_ = Position_;

} // namespace openmc

#endif // OPENMC_POSITION_H
