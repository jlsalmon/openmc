#ifndef OPENMC_ANGLE_ENERGY_H
#define OPENMC_ANGLE_ENERGY_H

namespace openmc {

//==============================================================================
//! Abstract type that defines a correlated or uncorrelated angle-energy
//! distribution that is a function of incoming energy. Each derived type must
//! implement a sample() method that returns an outgoing energy and
//! scattering cosine given an incoming energy.
//==============================================================================

class AngleEnergy {
public:
  virtual void sample(double E_in, double& E_out, double& mu) const = 0;
  virtual ~AngleEnergy() = default;
};

struct AngleEnergy_ {
  enum class Type {
    uncorrelated = 1, correlated = 2, nbody = 3, kalbach_mann = 3
  };
};

}

#endif // OPENMC_ANGLE_ENERGY_H
