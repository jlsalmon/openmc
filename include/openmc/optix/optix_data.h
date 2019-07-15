#ifndef OPENMC_OPTIX_DATA_H
#define OPENMC_OPTIX_DATA_H

#include <optix_world.h>

#include "openmc/angle_energy.h"
#include "openmc/secondary_uncorrelated.h"
#include "openmc/secondary_kalbach.h"
#include "openmc/distribution_angle.h"
#include "openmc/distribution_energy.h"

using namespace optix;
using namespace openmc;

void initialize_device_data();
Buffer initialize_nuclide_reactions(Context context, std::vector<Reaction*> reactions_);
void initialize_tabulated_1d(Tabulated1D_& tabulated_1d_, Context context, const Tabulated1D& tabulated_1d);
void initialize_polynomial(Polynomial_& polynomial_, Context context, const Polynomial& polynomial);
void initialize_angle_distribution(AngleDistribution_& angle_, Context context, AngleDistribution& angle);
void initialize_continuous_tabular(ContinuousTabular_& energy_, Context context, ContinuousTabular* ct);
void initialize_discrete_photon(DiscretePhoton_& energy_, Context context, DiscretePhoton* dp);
void initialize_level_inelastic(LevelInelastic_& energy_, Context context, LevelInelastic* li);

#endif //OPENMC_OPTIX_DATA_H
