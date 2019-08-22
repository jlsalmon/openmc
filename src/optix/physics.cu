#pragma once

#include <optix_world.h>

#include "variables.cu"
#include "nuclide.cu"
#include "reaction_product.cu"
#include "math_functions.cu"
#include "random_lcg.cu"

#include "openmc/particle.h"
#include "openmc/secondary_uncorrelated.h"

using namespace optix;
using namespace openmc;


__device__ __forceinline__
void _create_secondary(Particle_& p, Direction_ u, float E, Particle::Type type)
{
  // simulation::secondary_bank.emplace_back();

  auto& bank {secondary_bank_buffer[launch_index]};
  bank.particle = type;
  bank.wgt = p.wgt_;
  bank.r = p.r();
  bank.u = u;
  // bank.E = settings::run_CE ? E : g_; // FIXME: fixed source mode
  bank.E = E;
}


__forceinline__ __device__
void _inelastic_scatter(const Nuclide_& nuc, const Reaction_& rx, Particle_& p)
{
  // printf("inelastic scatter\n");
  // copy energy of neutron
  float E_in = p.E_;

  // sample outgoing energy and scattering cosine
  float E;
  float mu;
  // rx.products_[0].sample(E_in, E, mu);
  // rtPrintf("p.E_ before sampling reaction product: %f\n", p.E_);
  _sample_reaction_product(rx.products_[0], E_in, E, mu);
  // rtPrintf("p.E_ after sampling reaction product: %f\n", p.E_);
  // rtPrintf("E before: %f\n", E);

  // if scattering system is in center-of-mass, transfer cosine of scattering
  // angle and outgoing energy from CM to LAB
  if (rx.scatter_in_cm_) {
    float E_cm = E;

    // determine outgoing energy in lab
    float A = nuc.awr_;
    E = E_cm + (E_in + 2.0f*mu*(A + 1.0f) * sqrtf(E_in*E_cm))
               / ((A + 1.0f)*(A + 1.0f));

    // determine outgoing angle in lab
    mu = mu*sqrtf(E_cm/E) + 1.0f/(A+1.0f) * sqrtf(E_in/E);
  }

  // rtPrintf("p.E_ after outgoing energy in lab: %f\n", p.E_);
  // rtPrintf("E_in: %f\n", E_in);
  // rtPrintf("E after: %f\n", E);

  // Because of floating-point roundoff, it may be possible for mu to be
  // outside of the range [-1,1). In these cases, we just set mu to exactly -1
  // or 1
  if (fabsf(mu) > 1.0f) mu = copysignf(1.0f, mu);

  // Set outgoing energy and scattering angle
  p.E_ = E;
  p.mu_ = mu;

  rtPrintf("Rotating angle\n");
  // change direction of particle
  p.u() = rotate_angle(p.u(), mu, nullptr);

  // evaluate yield
  // double yield = (*rx->products_[0].yield_)(E_in); // FIXME: yield: is this the right distro?
  float yield;
  if (rx.products_[0].is_polynomial_yield) {
    rtPrintf("Polynomial yield\n");
    yield = _polynomial(rx.products_[0].polynomial_yield_, E_in);
  } else {
    rtPrintf("Tabulated 1D yield\n");
    yield = _tabulated_1d(rx.products_[0].tabulated_1d_yield_, E_in);
  }
  if (floorf(yield) == yield) {
    // If yield is integral, create exactly that many secondary particles
    for (int i = 0; i < static_cast<int>(roundf(yield)) - 1; ++i) {
      rtPrintf("Creating secondary particle\n");
      _create_secondary(p, p.u(), p.E_, Particle::Type::neutron);
    }
  } else {
    // Otherwise, change weight of particle based on yield
    p.wgt_ *= yield;
  }
  // printf("inelastic scatter done\n");
}


// __device__ __forceinline__
// Direction _sample_cxs_target_velocity(double awr, double E, Direction u, double kT)
// {
//   double beta_vn = std::sqrt(awr * E / kT);
//   double alpha = 1.0/(1.0 + std::sqrt(PI)*beta_vn/2.0);
//
//   double beta_vt_sq;
//   double mu;
//   while (true) {
//     // Sample two random numbers
//     double r1 = prn();
//     double r2 = prn();
//
//     if (prn() < alpha) {
//       // With probability alpha, we sample the distribution p(y) =
//       // y*e^(-y). This can be done with sampling scheme C45 frmo the Monte
//       // Carlo sampler
//
//       beta_vt_sq = -std::log(r1*r2);
//
//     } else {
//       // With probability 1-alpha, we sample the distribution p(y) = y^2 *
//       // e^(-y^2). This can be done with sampling scheme C61 from the Monte
//       // Carlo sampler
//
//       double c = std::cos(PI/2.0 * prn());
//       beta_vt_sq = -std::log(r1) - std::log(r2)*c*c;
//     }
//
//     // Determine beta * vt
//     double beta_vt = std::sqrt(beta_vt_sq);
//
//     // Sample cosine of angle between neutron and target velocity
//     mu = 2.0*prn() - 1.0;
//
//     // Determine rejection probability
//     double accept_prob = std::sqrt(beta_vn*beta_vn + beta_vt_sq -
//                                    2*beta_vn*beta_vt*mu) / (beta_vn + beta_vt);
//
//     // Perform rejection sampling on vt and mu
//     if (prn() < accept_prob) break;
//   }
//
//   // Determine speed of target nucleus
//   double vt = std::sqrt(beta_vt_sq*kT/awr);
//
//   // Determine velocity vector of target nucleus based on neutron's velocity
//   // and the sampled angle between them
//   return vt * rotate_angle(u, mu, nullptr);
// }


__device__ __forceinline__
Direction_ _sample_target_velocity(const Nuclide_& nuc, float E, Direction_ u,
                                 Direction_ v_neut, float xs_eff, float kT)
{
  // // check if nuclide is a resonant scatterer
  // ResScatMethod sampling_method;
  // if (nuc->resonant_) {
  //
  //   // sampling method to use
  //   sampling_method = settings::res_scat_method;
  //
  //   // upper resonance scattering energy bound (target is at rest above this E)
  //   if (E > settings::res_scat_energy_max) {
  //     return {};
  //
  //     // lower resonance scattering energy bound (should be no resonances below)
  //   } else if (E < settings::res_scat_energy_min) {
  //     sampling_method = ResScatMethod::cxs;
  //   }
  //
  //   // otherwise, use free gas model
  // } else {
  //   if (E >= FREE_GAS_THRESHOLD * kT && nuc->awr_ > 1.0) {
  //     return {};
  //   } else {
  //     sampling_method = ResScatMethod::cxs;
  //   }
  // }
  //
  // // use appropriate target velocity sampling method
  // switch (sampling_method) {
  //   case ResScatMethod::cxs:
  //
  //     // sample target velocity with the constant cross section (cxs) approx.
  //     return _sample_cxs_target_velocity(nuc->awr_, E, u, kT);
  //
  //   case ResScatMethod::dbrc:
  //   case ResScatMethod::rvs: {
  //     double E_red = std::sqrt(nuc->awr_ * E / kT);
  //     double E_low = std::pow(std::max(0.0, E_red - 4.0), 2) * kT / nuc->awr_;
  //     double E_up = (E_red + 4.0)*(E_red + 4.0) * kT / nuc->awr_;
  //
  //     // find lower and upper energy bound indices
  //     // lower index
  //     int i_E_low;
  //     if (E_low < nuc->energy_0K_.front()) {
  //       i_E_low = 0;
  //     } else if (E_low > nuc->energy_0K_.back()) {
  //       i_E_low = nuc->energy_0K_.size() - 2;
  //     } else {
  //       i_E_low = lower_bound_index(nuc->energy_0K_.begin(),
  //                                   nuc->energy_0K_.end(), E_low);
  //     }
  //
  //     // upper index
  //     int i_E_up;
  //     if (E_up < nuc->energy_0K_.front()) {
  //       i_E_up = 0;
  //     } else if (E_up > nuc->energy_0K_.back()) {
  //       i_E_up = nuc->energy_0K_.size() - 2;
  //     } else {
  //       i_E_up = lower_bound_index(nuc->energy_0K_.begin(),
  //                                  nuc->energy_0K_.end(), E_up);
  //     }
  //
  //     if (i_E_up == i_E_low) {
  //       // Handle degenerate case -- if the upper/lower bounds occur for the same
  //       // index, then using cxs is probably a good approximation
  //       return _sample_cxs_target_velocity(nuc->awr_, E, u, kT);
  //     }
  //
  //     if (sampling_method == ResScatMethod::dbrc) {
  //       // interpolate xs since we're not exactly at the energy indices
  //       double xs_low = nuc->elastic_0K_[i_E_low];
  //       double m = (nuc->elastic_0K_[i_E_low + 1] - xs_low)
  //                  / (nuc->energy_0K_[i_E_low + 1] - nuc->energy_0K_[i_E_low]);
  //       xs_low += m * (E_low - nuc->energy_0K_[i_E_low]);
  //       double xs_up = nuc->elastic_0K_[i_E_up];
  //       m = (nuc->elastic_0K_[i_E_up + 1] - xs_up)
  //           / (nuc->energy_0K_[i_E_up + 1] - nuc->energy_0K_[i_E_up]);
  //       xs_up += m * (E_up - nuc->energy_0K_[i_E_up]);
  //
  //       // get max 0K xs value over range of practical relative energies
  //       double xs_max = *std::max_element(&nuc->elastic_0K_[i_E_low + 1],
  //                                         &nuc->elastic_0K_[i_E_up + 1]);
  //       xs_max = std::max({xs_low, xs_max, xs_up});
  //
  //       while (true) {
  //         double E_rel;
  //         Direction v_target;
  //         while (true) {
  //           // sample target velocity with the constant cross section (cxs) approx.
  //           v_target = _sample_cxs_target_velocity(nuc->awr_, E, u, kT);
  //           Direction v_rel = v_neut - v_target;
  //           E_rel = v_rel.dot(v_rel);
  //           if (E_rel < E_up) break;
  //         }
  //
  //         // perform Doppler broadening rejection correction (dbrc)
  //         double xs_0K = nuc->elastic_xs_0K(E_rel);
  //         double R = xs_0K / xs_max;
  //         if (prn() < R) return v_target;
  //       }
  //
  //     } else if (sampling_method == ResScatMethod::rvs) {
  //       // interpolate xs CDF since we're not exactly at the energy indices
  //       // cdf value at lower bound attainable energy
  //       double m = (nuc->xs_cdf_[i_E_low] - nuc->xs_cdf_[i_E_low - 1])
  //                  / (nuc->energy_0K_[i_E_low + 1] - nuc->energy_0K_[i_E_low]);
  //       double cdf_low = nuc->xs_cdf_[i_E_low - 1]
  //                        + m * (E_low - nuc->energy_0K_[i_E_low]);
  //       if (E_low <= nuc->energy_0K_.front()) cdf_low = 0.0;
  //
  //       // cdf value at upper bound attainable energy
  //       m = (nuc->xs_cdf_[i_E_up] - nuc->xs_cdf_[i_E_up - 1])
  //           / (nuc->energy_0K_[i_E_up + 1] - nuc->energy_0K_[i_E_up]);
  //       double cdf_up = nuc->xs_cdf_[i_E_up - 1]
  //                       + m*(E_up - nuc->energy_0K_[i_E_up]);
  //
  //       while (true) {
  //         // directly sample Maxwellian
  //         double E_t = -kT * std::log(prn());
  //
  //         // sample a relative energy using the xs cdf
  //         double cdf_rel = cdf_low + prn()*(cdf_up - cdf_low);
  //         int i_E_rel = lower_bound_index(&nuc->xs_cdf_[i_E_low-1],
  //                                         &nuc->xs_cdf_[i_E_up+1], cdf_rel);
  //         double E_rel = nuc->energy_0K_[i_E_low + i_E_rel];
  //         double m = (nuc->xs_cdf_[i_E_low + i_E_rel]
  //                     - nuc->xs_cdf_[i_E_low + i_E_rel - 1])
  //                    / (nuc->energy_0K_[i_E_low + i_E_rel + 1]
  //                       -  nuc->energy_0K_[i_E_low + i_E_rel]);
  //         E_rel += (cdf_rel - nuc->xs_cdf_[i_E_low + i_E_rel - 1]) / m;
  //
  //         // perform rejection sampling on cosine between
  //         // neutron and target velocities
  //         double mu = (E_t + nuc->awr_ * (E - E_rel)) /
  //                     (2.0 * std::sqrt(nuc->awr_ * E * E_t));
  //
  //         if (std::abs(mu) < 1.0) {
  //           // set and accept target velocity
  //           E_t /= nuc->awr_;
  //           return std::sqrt(E_t) * rotate_angle(u, mu, nullptr);
  //         }
  //       }
  //     }
  //   } // case RVS, DBRC
  // } // switch (sampling_method)

  // UNREACHABLE();
}


__device__ __forceinline__
void _elastic_scatter(int i_nuclide, const Reaction_& rx, float kT, Particle_& p)
{
  // get pointer to nuclide
  const auto& nuc = nuclide; // FIXME: {data::nuclides[i_nuclide]};

  float vel = sqrtf(p.E_);
  float awr = nuc.awr_;

  // Neutron velocity in LAB
  Direction_ v_n = vel*p.u();

  // Sample velocity of target nucleus
  Direction_ v_t {};
  if (!p.neutron_xs_[i_nuclide].use_ptable) {
    v_t = _sample_target_velocity(nuc, p.E_, p.u(), v_n,
                                 p.neutron_xs_[i_nuclide].elastic, kT);
  }

  // Velocity of center-of-mass
  Direction_ v_cm = (v_n + awr*v_t)/(awr + 1.0f);

  // Transform to CM frame
  v_n -= v_cm;

  // Find speed of neutron in CM
  vel = v_n.norm();

  // Sample scattering angle, checking if it is an ncorrelated angle-energy
  // distribution
  float mu_cm;

  auto &product = rx.products_[0];
  // auto d_ = (UncorrelatedAngleEnergy_&) d;

  // ReactionProduct_& product = rx.products_[0];
  // printf("products_[0].distribution_type: %d\n", product.distribution_type);
  // printf("products_[0].distribution_.size: %lu\n", product.distribution_.size());
  // printf("products_[0].distribution_[0].fission_: %d\n", product.distribution_[0].fission_);
  //
  // printf("products_[0].distribution_[0].angle_empty: %d\n", product.distribution_[0].angle_empty);
  // printf("products_[0].distribution_[0].angle_.distribution_.id: %i\n", product.distribution_[0].angle_.distribution_.getId());
  // printf("products_[0].distribution_[0].angle_.distribution_.size: %lu\n", product.distribution_[0].angle_.distribution_.size());
  // printf("products_[0].distribution_[0].angle_.energy_size: %lu\n", product.distribution_[0].angle_.energy_size);
  // printf("products_[0].distribution_[0].angle_.energy_.id: %i\n", product.distribution_[0].angle_.energy_.getId());
  //
  // printf("products_[0].distribution_[0].energy_.type: %d\n", product.distribution_[0].energy_type);
  // printf("products_[0].distribution_[0].energy_.energy_size: %lu\n", product.distribution_[0].energy_.energy_size);
  // printf("products_[0].distribution_[0].energy_.energy_.id: %i\n", product.distribution_[0].energy_.energy_.getId());
  // printf("products_[0].distribution_[0].energy_.distribution_.id: %d\n", product.distribution_[0].energy_.distribution_.getId());

  // const UncorrelatedAngleEnergy_& d = dynamic_cast<const UncorrelatedAngleEnergy_ &>(rx.products_[0].distribution_[0]);
  // auto d_ = dynamic_cast<UncorrelatedAngleEnergy*>(d.get());
  // printf("LOL...\n");
  // printf("LOL %lu\n", d.angle_.distribution_.size());
  // printf("LOL %i\n", d.angle_.energy_.getId());
  // printf("LOL %lu\n", d.angle_.energy_.size());
  if (product.distribution_type == AngleEnergy_::Type::uncorrelated) { // FIXME
    auto& d = rx.products_[0].distribution_uae[0];
    // mu_cm = d.angle().sample(p.E_);
    rtPrintf("Sampling angle dist from _elastic_scatter\n");
    mu_cm = _sample_angle_distribution(d.angle_, p.E_);
  } else {
    mu_cm = 2.0f*prn() - 1.0f;
  }

  // Determine direction cosines in CM
  Direction_ u_cm = v_n/vel;

  // Rotate neutron velocity vector to new angle -- note that the speed of the
  // neutron in CM does not change in elastic scattering. However, the speed
  // will change when we convert back to LAB
  v_n = vel * rotate_angle(u_cm, mu_cm, nullptr);

  // Transform back to LAB frame
  v_n += v_cm;

  p.E_ = v_n.dot(v_n);
  vel = sqrtf(p.E_);

  // compute cosine of scattering angle in LAB frame by taking dot product of
  // neutron's pre- and post-collision angle
  p.mu_ = p.u().dot(v_n) / vel;

  // Set energy and direction of particle in LAB frame
  p.u() = v_n / vel;

  // Because of floating-point roundoff, it may be possible for mu_lab to be
  // outside of the range [-1,1). In these cases, we just set mu_lab to exactly
  // -1 or 1
  if (fabsf(p.mu_) > 1.0f) p.mu_ = copysignf(1.0f, p.mu_);
}


// __device__ __forceinline__
// void _sab_scatter(int i_nuclide, int i_sab, Particle_& p)
// {
//   // Determine temperature index
//   const auto& micro {p.neutron_xs_[i_nuclide]};
//   int i_temp = micro.index_temp_sab;
//
//   // Sample energy and angle
//   double E_out;
//   data::thermal_scatt[i_sab]->data_[i_temp].sample(micro, p.E_, &E_out, &p.mu_);
//
//   // Set energy to outgoing, change direction of particle
//   p.E_ = E_out;
//   p.u() = rotate_angle(p.u(), p.mu_, nullptr);
// }



__device__ __forceinline__
void _scatter(Particle_& p, int i_nuclide)
{
  // copy incoming direction
  Direction_ u_old {p.u()};

  // Get pointer to nuclide and grid index/interpolation factor
  const auto& nuc = nuclide; // FIXME: {data::nuclides[i_nuclide]};
  const auto& micro {p.neutron_xs_[i_nuclide]};
  int i_temp =  micro.index_temp;
  int i_grid =  micro.index_grid;
  float f = micro.interp_factor;

  // For tallying purposes, this routine might be called directly. In that
  // case, we need to sample a reaction via the cutoff variable
  float cutoff = prn() * (micro.total - micro.absorption);
  bool sampled = false;

  // Calculate elastic cross section if it wasn't precalculated
  if (micro.elastic == CACHE_INVALID) {
    // rtPrintf("CALCULATING ELASTIC XS\n");
    // rtPrintf("micro.elastic before: %lf\n", micro.elastic);
    _calculate_elastic_xs(nuc, p);
    // rtPrintf("micro.elastic after: %lf\n", micro.elastic);
  }

  // printf("\n");
  // printf("micro.total: %lf\n", micro.total);
  // printf("micro.absorption: %lf\n", micro.absorption);
  // printf("micro.fission: %lf\n", micro.fission);
  // printf("micro.nu_fission: %lf\n", micro.nu_fission);
  // printf("micro.elastic: %lf\n", micro.elastic);
  // printf("micro.thermal: %lf\n", micro.thermal);
  // printf("micro.thermal_elastic: %lf\n", micro.thermal_elastic);
  // printf("micro.photon_prod: %lf\n", micro.photon_prod);
  // printf("micro.index_grid: %d\n", micro.index_grid);
  // printf("micro.index_temp: %d\n", micro.index_temp);
  // printf("micro.interp_factor: %lf\n", micro.interp_factor);
  // printf("micro.index_sab: %d\n", micro.index_sab);
  // printf("micro.index_temp_sab: %d\n", micro.index_temp_sab);
  // printf("micro.sab_frac: %lf\n", micro.sab_frac);
  // printf("micro.last_E: %lf\n", micro.last_E);
  // printf("micro.last_sqrtkT: %lf\n", micro.last_sqrtkT);
  // printf("\n");
  //
  // rtPrintf("cutoff: %lf\n", cutoff);
  // rtPrintf("i_temp: %i\n", i_temp);
  // rtPrintf("i_grid: %i\n", i_grid);

  float prob = micro.elastic - micro.thermal;
  // rtPrintf("prob: %lf\n", prob);

  if (prob > cutoff) {
    // =======================================================================
    // NON-S(A,B) ELASTIC SCATTERING

    // Determine temperature
    // double kT = nuc.multipole_ ? p.sqrtkT_*p.sqrtkT_ : nuc.kTs_[i_temp]; // FIXME: multipole
    float kT = nuc.kTs_[i_temp];

    // Perform collision physics for elastic scattering
    _elastic_scatter(i_nuclide, nuc.reactions_[0], kT, p);
    p.event_mt_ = ELASTIC;
    sampled = true;
  }

  prob = micro.elastic;
  if (prob > cutoff && !sampled) { // FIXME: sab
    printf("FIXME: SAB SCATTERING\n");
  //   // =======================================================================
  //   // S(A,B) SCATTERING
  //
  //   _sab_scatter(i_nuclide, micro.index_sab, p);
  //
  //   p.event_mt_ = ELASTIC;
  //   sampled = true;
  }

  if (!sampled) {
    // =======================================================================
    // INELASTIC SCATTERING

    int j = 0;
    int i = 0;
    while (prob < cutoff) {
      i = nuc.index_inelastic_scatter_[j];
      ++j;

      // rtPrintf("i: %d j: %d\n", i, j);

      // Check to make sure inelastic scattering reaction sampled
      if (i >= nuc.reactions_.size()) {
        // p.write_restart();
        // fatal_error("Did not sample any reaction for nuclide " + nuc->name_);
        printf("ERROR: Did not sample any reaction for nuclide FIXME\n");
      }

      // if energy is below threshold for this reaction, skip it
      // const auto& xs {nuc.reactions_[i].xs_[i_temp]};
      const auto& xs {nuc.reactions_[i].xs_[i_temp]};
      if (i_grid < xs.threshold) {
        // rtPrintf("i_grid %i less than threshold %i\n", i_grid, xs.threshold);
        continue;
      }

      // rtPrintf("nuc.reactions_[i]: %p\n", nuc.reactions_[i]);
      // rtPrintf("nuc.reactions_[i].xs_[i_temp]: %p\n", nuc.reactions_[i].xs_[i_temp]);
      // rtPrintf("xs.value[i_grid - xs.threshold]: %lf\n", xs.value_[i_grid - xs.threshold]);
      // rtPrintf("xs.value[i_grid - xs.threshold + 1]: %lf\n", xs.value_[i_grid - xs.threshold + 1]);

      // add to cumulative probability
      prob += (1.0f - f)*xs.value_[i_grid - xs.threshold] +
              f*xs.value_[i_grid - xs.threshold + 1];

      // rtPrintf("prob is now: %lf\n", prob);
    }

    // Perform collision physics for inelastic scattering
    const auto& rx {nuc.reactions_[i]};
    _inelastic_scatter(nuc, rx, p);
    p.event_mt_ = rx.mt_;
  }

  // Set event component
  p.event_ = EVENT_SCATTER;

  // Sample new outgoing angle for isotropic-in-lab scattering
  const auto& mat = material_buffer[0]; // FIXME: {model::materials[p.material_]};
  // if (!mat->p0_.empty()) { // FIXME: isotropic-in-lab
  //   int i_nuc_mat = mat->mat_nuclide_index_[i_nuclide];
  //   if (mat->p0_[i_nuc_mat]) {
  //     // Sample isotropic-in-lab outgoing direction
  //     double mu = 2.0*prn() - 1.0;
  //     double phi = 2.0*PI*prn();
  //
  //     // Change direction of particle
  //     p.u().x = mu;
  //     p.u().y = std::sqrt(1.0 - mu*mu)*std::cos(phi);
  //     p.u().z = std::sqrt(1.0 - mu*mu)*std::sin(phi);
  //     p.mu_ = u_old.dot(p.u());
  //   }
  // }
}


__device__ __forceinline__
void _absorption(Particle_& p, int i_nuclide)
{
  // if (settings::survival_biasing) { // FIXME: survival biasing
  //   // Determine weight absorbed in survival biasing
  //   p.wgt_absorb_ = p.wgt_ * p.neutron_xs_[i_nuclide].absorption /
  //                    p.neutron_xs_[i_nuclide].total;
  //
  //   // Adjust weight of particle by probability of absorption
  //   p.wgt_ -= p.wgt_absorb_;
  //   p.wgt_last_ = p.wgt_;
  //
  //   // Score implicit absorption estimate of keff
  //   if (settings::run_mode == RUN_MODE_EIGENVALUE) {
  //     global_tally_absorption += p.wgt_absorb_ * p.neutron_xs_[
  //       i_nuclide].nu_fission / p.neutron_xs_[i_nuclide].absorption;
  //   }
  // } else {
    // See if disappearance reaction happens
    if (p.neutron_xs_[i_nuclide].absorption >
        prn() * p.neutron_xs_[i_nuclide].total) {
      // rtPrintf("Absorbed\n");
      // Score absorption estimate of keff
      // if (settings::run_mode == RUN_MODE_EIGENVALUE) { // FIXME: run modes
        global_tally_absorption_buffer[launch_index] += p.wgt_ * p.neutron_xs_[
          i_nuclide].nu_fission / p.neutron_xs_[i_nuclide].absorption;
      // }

      p.alive_ = false;
      p.event_ = EVENT_ABSORB;
      p.event_mt_ = N_DISAPPEAR;
    }
  // }
}


__device__ __forceinline__
void _sample_fission_neutron(int i_nuclide, const Reaction_& rx, float E_in, Particle_::Bank_& site)
{
  // Sample cosine of angle -- fission neutrons are always emitted
  // isotropically. Sometimes in ACE data, fission reactions actually have
  // an angular distribution listed, but for those that do, it's simply just
  // a uniform distribution in mu
  float mu = 2.0f * prn() - 1.0f;
  // Sample azimuthal angle uniformly in [0,2*pi)
  float phi = 2.0f*M_PIf*prn();

  site.u.x = mu;
  site.u.y = sqrtf(1.0f - mu*mu) * cosf(phi);
  site.u.z = sqrtf(1.0f - mu*mu) * sinf(phi);

  // Determine total nu, delayed nu, and delayed neutron fraction
  const auto& nuc = nuclide; // FIXME: {data::nuclides[i_nuclide]};
  float nu_t = _nu(nuc, E_in, Nuclide::EmissionMode::total);
  float nu_d = _nu(nuc, E_in, Nuclide::EmissionMode::delayed);
  float beta = nu_d / nu_t;

  if (prn() < beta) {
    rtPrintf("Delayed neutron sampled\n");
    // ====================================================================
    // DELAYED NEUTRON SAMPLED

    // sampled delayed precursor group
    float xi = prn()*nu_d; // FIXME
    float prob = 0.0f;
    int group;
    for (group = 1; group < nuc.n_precursor_; ++group) {
      // determine delayed neutron precursor yield for group j
      // float yield = (rx.products_[group].yield_)(E_in);
      float yield;
      if (rx.products_[group].is_polynomial_yield) {
        yield = _polynomial(rx.products_[group].polynomial_yield_, E_in);
      } else {
        yield = _tabulated_1d(rx.products_[group].tabulated_1d_yield_, E_in);
      }


      // Check if this group is sampled
      prob += yield;
      if (xi < prob) break;
    }

    // if the sum of the probabilities is slightly less than one and the
    // random number is greater, j will be greater than nuc %
    // n_precursor -- check for this condition
    group = min(group, nuc.n_precursor_);

    // set the delayed group for the particle born from fission
    site.delayed_group = group;

    int n_sample = 0;
    while (true) {
      // sample from energy/angle distribution -- note that mu has already been
      // sampled above and doesn't need to be resampled
      // rx.products_[group].sample(E_in, site.E, mu);
      _sample_reaction_product(rx.products_[group], E_in, site.E, mu);

      // resample if energy is greater than maximum neutron energy
      constexpr int neutron = static_cast<int>(Particle::Type::neutron);
      if (site.E < energy_max_neutron) break;

      // check for large number of resamples
      ++n_sample;
      if (n_sample == MAX_SAMPLE) {
        // particle_write_restart(p)
        // fatal_error("Resampled energy distribution maximum number of times "
        //             "for nuclide " + nuc.name_);
        printf("ERROR: Resampled energy distribution maximum number of times for nuclide FIXME\n");
      }
    }

  } else {
    rtPrintf("Prompt neutron sampled\n");
    // ====================================================================
    // PROMPT NEUTRON SAMPLED

    // set the delayed group for the particle born from fission to 0
    site.delayed_group = 0;

    // sample from prompt neutron energy distribution
    int n_sample = 0;
    while (true) {
      _sample_reaction_product(rx.products_[0], E_in, site.E, mu);

      // resample if energy is greater than maximum neutron energy
      constexpr int neutron = static_cast<int>(Particle::Type::neutron);
      if (site.E < energy_max_neutron) break;

      // check for large number of resamples
      ++n_sample;
      if (n_sample == MAX_SAMPLE) {
        // particle_write_restart(p)
        // fatal_error("Resampled energy distribution maximum number of times "
        //             "for nuclide " + nuc.name_);
        printf("ERROR: Resampled energy distribution maximum number of times for nuclide FIXME\n");
      }
    }
  }
}


__device__ __forceinline__
void _create_fission_sites(Particle_& p, int i_nuclide, const Reaction_& rx)
{
  // If uniform fission source weighting is turned on, we increase or decrease
  // the expected number of fission sites produced
  float weight = /*settings::ufs_on ? ufs_get_weight(p) :*/ 1.0f; // FIXME: ufs

  // Determine the expected number of neutrons produced
  float nu_t = p.wgt_ / _simulation[0].keff * weight * p.neutron_xs_[
    i_nuclide].nu_fission / p.neutron_xs_[i_nuclide].total;

  // Sample the number of neutrons produced
  int nu = static_cast<int>(nu_t);
  if (prn() <= (nu_t - nu)) ++nu;


  // Begin banking the source neutrons
  // First, if our bank is full then don't continue
  if (nu == 0) return;

  // Initialize the counter of delayed neutrons encountered for each delayed
  // group.
  float nu_d[MAX_DELAYED_GROUPS] = {0.f};

  // printf("bank size: %lu nu: %d nu_t: %f, keff: %f\n", fission_bank_buffer.size(), nu, nu_t, _simulation[0].keff);
  // printf("p.wgt_: %lf, weight: %lf, p->neutron_xs_[i_nuclide].nu_fission: %lf, p->neutron_xs_[i_nuclide].total: %lf \n",
  //        p.wgt_, weight, p.neutron_xs_[i_nuclide].nu_fission, p.neutron_xs_[i_nuclide].total);

  p.fission_ = true;
  for (int i = 0; i < nu; ++i) {
    // Create new bank site and get reference to last element
    // fission_bank.emplace_back();
    // auto& site {bank.back()};
    // printf("fission bank size: %lu index: %d nu: %d keff: %f\n", fission_bank_buffer.size(), (3 * launch_index) + i, nu, keff);
    auto &site = fission_bank_buffer[(3 * launch_index) + i];

    // Bank source neutrons by copying the particle data
    site.r = p.r();
    site.particle = Particle::Type::neutron;
    site.wgt = 1.f / weight;

    // printf("site.wgt: %lf\n", site.wgt);

    // Sample delayed group and angle/energy for fission reaction
    _sample_fission_neutron(i_nuclide, rx, p.E_, site);

    // Set the delayed group on the particle as well
    p.delayed_group_ = site.delayed_group;

    // Increment the number of neutrons born delayed
    if (p.delayed_group_ > 0) {
      nu_d[p.delayed_group_-1]++;
    }
  }

  // Store the total weight banked for analog fission tallies
  p.n_bank_ = nu;
  p.wgt_bank_ = nu / weight;
  for (size_t d = 0; d < MAX_DELAYED_GROUPS; d++) {
    p.n_delayed_bank_[d] = nu_d[d];
  }
}


__device__ __forceinline__
const Reaction_& _sample_fission(int i_nuclide, const Particle_& p)
{
  // Get pointer to nuclide
  const auto& nuc = nuclide; // FIXME: {data::nuclides[i_nuclide]};

  // If we're in the URR, by default use the first fission reaction. We also
  // default to the first reaction if we know that there are no partial fission
  // reactions
  // if (p.neutron_xs_[i_nuclide].use_ptable || !nuc.has_partial_fission_) {
    return nuc.fission_rx_[0]; // FIXME: support more than one reaction
  // }

  // // Check to see if we are in a windowed multipole range.  WMP only supports
  // // the first fission reaction.
  // if (nuc->multipole_) { // FIXME: multipole
  //   if (p.E_ >= nuc->multipole_->E_min_ && p.E_ <= nuc->multipole_->E_max_) {
  //     return nuc->fission_rx_[0];
  //   }
  // }

  // // Get grid index and interpolatoin factor and sample fission cdf
  // int i_temp = p.neutron_xs_[i_nuclide].index_temp;
  // int i_grid = p.neutron_xs_[i_nuclide].index_grid;
  // double f = p.neutron_xs_[i_nuclide].interp_factor;
  // double cutoff = prn() * p.neutron_xs_[i_nuclide].fission;
  // double prob = 0.0;
  //
  // // Loop through each partial fission reaction type
  // for (auto& rx : nuc->fission_rx_) {
  //   // if energy is below threshold for this reaction, skip it
  //   int threshold = rx->xs_[i_temp].threshold;
  //   if (i_grid < threshold) continue;
  //
  //   // add to cumulative probability
  //   prob += (1.0 - f) * rx->xs_[i_temp].value[i_grid - threshold]
  //           + f*rx->xs_[i_temp].value[i_grid - threshold + 1];
  //
  //   // Create fission bank sites if fission occurs
  //   if (prob > cutoff) return rx;
  // }
  //
  // // If we reached here, no reaction was sampled
  // throw std::runtime_error{"No fission reaction was sampled for " + nuc->name_};
}


__device__ __forceinline__
int _sample_nuclide(const Particle_& p)
{
  // printf("Sampling nuclide\n");

  // Sample cumulative distribution function
  float cutoff = prn() * p.macro_xs_.total;

  // Get pointers to nuclide/density arrays
  const auto& mat = material_buffer[0]; // FIXME: {model::materials[p.material_]};
  int n = 1; // FIXME: mat->nuclide_.size();

  float prob = 0.0f;
  for (int i = 0; i < n; ++i) {
    // Get atom density
    int i_nuclide = mat.nuclide_[i];
    float atom_density = mat.atom_density_[i];

    // Increment probability to compare to cutoff
    prob += atom_density * p.neutron_xs_[i_nuclide].total;
    if (prob >= cutoff) return i_nuclide;
  }

  // If we reach here, no nuclide was sampled
  // p.write_restart(); // FIXME: particle restart
  // throw std::runtime_error{"Did not sample any nuclide during collision."};
  rtPrintf("ERROR: Did not sample any nuclide during collision at index %d.\n", launch_index);
  rtPrintf("cutoff: %f\n", cutoff);
  rtPrintf("p.macro_xs_.total: %f\n", p.macro_xs_.total);
  rtPrintf("p.neutron_xs_[i_nuclide].total: %f\n", p.neutron_xs_[0].total);
  rtPrintf("prob: %f\n", prob);
  rtPrintf("atom_density: %f\n", mat.atom_density_[0]);
}


__device__ __forceinline__
void _sample_neutron_reaction(Particle_& p)
{
  // Sample a nuclide within the material
  int i_nuclide = _sample_nuclide(p);

  // Save which nuclide particle had collision with
  p.event_nuclide_ = i_nuclide;

  // Create fission bank sites. Note that while a fission reaction is sampled,
  // it never actually "happens", i.e. the weight of the particle does not
  // change when sampling fission sites. The following block handles all
  // absorption (including fission)

  const auto& nuc = nuclide; // {data::nuclides[i_nuclide]}; FIXME: support multiple nuclides

  if (nuc.fissionable_) {
    const Reaction_& rx = _sample_fission(i_nuclide, p);
    // // if (settings::run_mode == RUN_MODE_EIGENVALUE) {
      _create_fission_sites(p, i_nuclide, rx);
    // } else if (settings::run_mode == RUN_MODE_FIXEDSOURCE && // FIXME: fixed source mode
    //            settings::create_fission_neutrons) {
    //   create_fission_sites(p, i_nuclide, rx, simulation::secondary_bank);
    //
    //   // Make sure particle population doesn't grow out of control for
    //   // subcritical multiplication problems.
    //   if (simulation::secondary_bank.size() >= 10000) {
    //     fatal_error("The secondary particle bank appears to be growing without "
    //                 "bound. You are likely running a subcritical multiplication problem "
    //                 "with k-effective close to or greater than one.");
    //   }
    // }
  }

  // // Create secondary photons // FIXME: photon support
  // if (settings::photon_transport) {
  //   prn_set_stream(STREAM_PHOTON);
  //   sample_secondary_photons(p, i_nuclide);
  //   prn_set_stream(STREAM_TRACKING);
  // }

  // If survival biasing is being used, the following subroutine adjusts the
  // weight of the particle. Otherwise, it checks to see if absorption occurs

  if (p.neutron_xs_[i_nuclide].absorption > 0.0f) {
    rtPrintf("Absorption\n");
    _absorption(p, i_nuclide);
  } else {
    p.wgt_absorb_ = 0.0f;
  }
  if (!p.alive_) return;

  // Sample a scattering reaction and determine the secondary energy of the
  // exiting neutron
  rtPrintf("Scatter\n");
  _scatter(p, i_nuclide);

  // Advance URR seed stream 'N' times after energy changes
  if (p.E_ != p.E_last_) {
    prn_set_stream(_STREAM_URR_PTABLE); // FIXME: random number generation
    advance_prn_seed(num_nuclides /* FIXME: data::nuclides.size()*/);
    prn_set_stream(_STREAM_TRACKING);
  }

  // // Play russian roulette if survival biasing is turned on // FIXME: survival biasing
  // if (settings::survival_biasing) {
  //   russian_roulette(p);
  //   if (!p.alive_) return;
  // }
}


__device__ __forceinline__
void _collision(Particle_& p)
{
  rtPrintf("collision\n");

  // Add to collision counter for particle
  ++(p.n_collision_);

  // Sample reaction for the material the particle is in
  // switch (static_cast<Particle::Type>(p.type_)) { // FIXME: other particle types
  //   case Particle::Type::neutron:
      _sample_neutron_reaction(p);
  //     break;
  //   case Particle::Type::photon:
  //     sample_photon_reaction(p);
  //     break;
  //   case Particle::Type::electron:
  //     sample_electron_reaction(p);
  //     break;
  //   case Particle::Type::positron:
  //     sample_positron_reaction(p);
  //     break;
  // }

  // Kill particle if energy falls below cutoff
  int type = static_cast<int>(p.type_);
  if (p.E_ < 0.0f/*FIXME: settings::energy_cutoff[type]*/) {
    p.alive_ = false;
    p.wgt_ = 0.0f;
  }

  // // Display information about collision
  // if (settings::verbosity >= 10 || simulation::trace) {
  //   std::stringstream msg;
  //   if (p.type_ == Particle::Type::neutron) {
  //     msg << "    " << reaction_name(p.event_mt_) << " with " <<
  //         data::nuclides[p.event_nuclide_]->name_ << ". Energy = " << p.E_ << " eV.";
  //   } else {
  //     msg << "    " << reaction_name(p.event_mt_) << " with " <<
  //         data::elements[p.event_nuclide_].name_ << ". Energy = " << p.E_ << " eV.";
  //   }
  //   write_message(msg, 1);
  // }

  rtPrintf("%s with %c%c%c%c. Energy = %lf eV.\n", "FIXME" /*FIXME: reaction_name(p.event_mt_)*/,
         nuclide.name_[0], nuclide.name_[1], nuclide.name_[2], nuclide.name_[3], p.E_);
}
