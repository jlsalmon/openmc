#include <cuda.h>
#include <openmc/distribution_spatial.h>
#include <openmc/source.h>

#include "openmc/settings.h"
#include "openmc/simulation.h"
#include "openmc/nuclide.h"
#include "openmc/material.h"
#include "openmc/thermal.h"
#include "openmc/particle.h"
#include "openmc/cell.h"
#include "openmc/random_lcg.h"
#include "openmc/angle_energy.h"
#include "openmc/secondary_uncorrelated.h"
#include "openmc/secondary_kalbach.h"
#include "openmc/distribution_angle.h"
#include "openmc/distribution_energy.h"

#include "openmc/optix/optix_geometry.h"
#include "openmc/optix/optix_data.h"

using namespace openmc;
using namespace optix;

void precompile_kernels() {
  Context context = geometry->context;
  printf("Precompiling OptiX kernels...\n");

  context->launch(0, 0);
  context->launch(1, 0);
  context->launch(2, 0);

  printf("Precompilation complete\n");
}

void initialize_device_data() {
  Context context = geometry->context;
  printf("Initialising device data...\n");

  // Cell buffer
  Cell_ cells[model::cells.size()];
  for (int i = 0; i < model::cells.size(); ++i) {
    cells[i] = Cell_(model::cells[i].get());
  }
  Buffer cell_buffer = context->createBuffer(RT_BUFFER_INPUT);
  cell_buffer->setFormat(RT_FORMAT_USER);
  cell_buffer->setElementSize(sizeof(Cell_));
  cell_buffer->setSize(model::cells.size() * sizeof(Cell_));
  memcpy(cell_buffer->map(), cells, model::cells.size() * sizeof(Cell_));
  cell_buffer->unmap();
  context["cell_buffer"]->set(cell_buffer);

  // ===========================================================================
  // Output buffers
  // ===========================================================================

  // TODO: figure out how big these buffers should be based on the number of
  //  particles and the amount of memory available, and split the transport into
  //  multiple launches if necessary

  // Source bank buffer
  Buffer source_bank_buffer = context->createBuffer(
    RT_BUFFER_OUTPUT, RT_FORMAT_USER, static_cast<RTsize>(simulation::work_per_rank));
  source_bank_buffer->setElementSize(sizeof(Particle_::Bank_));
  context["source_bank_buffer"]->set(source_bank_buffer);

  // Fission bank buffer
  Buffer fission_bank_buffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_USER);
  fission_bank_buffer->setElementSize(sizeof(Particle_::Bank_));
  fission_bank_buffer->setSize(static_cast<RTsize>(3 * simulation::work_per_rank * 2)); // FIXME: is this the right size?
  context["fission_bank_buffer"]->set(fission_bank_buffer);

  // Secondary bank buffer
  Buffer secondary_bank_buffer = context->createBuffer(
    RT_BUFFER_OUTPUT, RT_FORMAT_USER, static_cast<RTsize>(simulation::work_per_rank));
  secondary_bank_buffer->setElementSize(sizeof(Particle_::Bank_));
  context["secondary_bank_buffer"]->set(secondary_bank_buffer);

  // ===========================================================================
  // Variables
  // ===========================================================================

  context["n_particles"]->setUint(settings::n_particles);
  context["total_gen"]->setInt(simulation::total_gen);
  context["num_nuclides"]->setUint(data::nuclides.size());
  context["log_spacing"]->setFloat(static_cast<float>(simulation::log_spacing));
  context["energy_min_neutron"]->setFloat(
    static_cast<float>(data::energy_min[static_cast<int>(Particle::Type::neutron)]));
  context["energy_max_neutron"]->setFloat(
    static_cast<float>(data::energy_max[static_cast<int>(Particle::Type::neutron)]));
  context["temperature_method"]->setInt(settings::temperature_method);

  // Material
  // FIXME: support more than one material
  openmc::Material *material = model::materials[0].get();
  Material_ material_(material);
  context["material"]->setUserData(sizeof(material_), &material_);

  // ===========================================================================
  // Input buffers
  // ===========================================================================

  // Particles
  Buffer particle_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
  particle_buffer->setElementSize(sizeof(Particle_));
  particle_buffer->setSize(simulation::work_per_rank);
  context["particle_buffer"]->setBuffer(particle_buffer);

  // External sources
  // FIXME: support more than one source and more than one spatial/angle/energy dist
  SourceDistribution &source = model::external_sources[0];
  SourceDistribution_ source_;

  SpatialBox_ space_;
  SpatialBox *space = dynamic_cast<SpatialBox *>(source.space_.get());
  space_.lower_left_ = Position_ {static_cast<float>(space->lower_left_.x), static_cast<float>(space->lower_left_.y), static_cast<float>(space->lower_left_.z)};
  space_.upper_right_ = Position_ {static_cast<float>(space->upper_right_.x), static_cast<float>(space->upper_right_.y), static_cast<float>(space->upper_right_.z)};
  space_.only_fissionable_ = space->only_fissionable_;

  Isotropic_ angle_;
  Isotropic *angle = dynamic_cast<Isotropic *>(source.angle_.get());
  angle_.u_ref_ = Direction_ {static_cast<float>(angle->u_ref_.x), static_cast<float>(angle->u_ref_.y), static_cast<float>(angle->u_ref_.z)};

  Watt_ energy_;
  Watt *energy = dynamic_cast<Watt *>(source.energy_.get());
  energy_.a_ = static_cast<float>(energy->a_);
  energy_.b_ = static_cast<float>(energy->b_);

  source_.space_ = space_;
  source_.angle_ = angle_;
  source_.energy_ = energy_;
  Buffer external_sources_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
  external_sources_buffer->setElementSize(sizeof(SourceDistribution_));
  external_sources_buffer->setSize(1);
  memcpy(external_sources_buffer->map(), &source_, 1 * sizeof(SourceDistribution_));
  context["external_sources_buffer"]->setBuffer(external_sources_buffer);
  external_sources_buffer->unmap();

  // ===========================================================================
  // Nuclide
  // ===========================================================================
  Nuclide *nuclide = data::nuclides[0].get();

  // Nuclide.kTs_
  Buffer kTs_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
  kTs_buffer->setElementSize(sizeof(float));
  kTs_buffer->setSize(nuclide->kTs_.size());
  std::vector<float> floats(nuclide->kTs_.size());
  std::transform(std::begin(nuclide->kTs_), std::end(nuclide->kTs_), std::begin(floats), [&](const double& value) { return static_cast<float>(value); });
  memcpy(kTs_buffer->map(), floats.data(), nuclide->kTs_.size() * sizeof(float));
  kTs_buffer->unmap();
  // printf("nuclide_.kTs_ buffer id: %d\n", kTs_buffer->getId());
  // printf("nuclide_.kTs_.size: %lu\n", nuclide->kTs_.size());

  // Nuclide.grid_
  std::vector<Nuclide_::EnergyGrid_> grids;
  for (auto &grid : nuclide->grid_) {

    // Energy.grid_index
    Buffer grid_index_buffer = context->createBuffer(RT_BUFFER_INPUT);
    grid_index_buffer->setFormat(RT_FORMAT_USER);
    grid_index_buffer->setElementSize(sizeof(int));
    grid_index_buffer->setSize(grid.grid_index.size());
    memcpy(grid_index_buffer->map(), grid.grid_index.data(), grid.grid_index.size() * sizeof(int));
    grid_index_buffer->unmap();
    // printf("EnergyGrid_.grid_index buffer id: %d\n", grid_index_buffer->getId());

    // EnergyGrid.energy
    Buffer energy_buffer = context->createBuffer(RT_BUFFER_INPUT);
    energy_buffer->setFormat(RT_FORMAT_USER);
    energy_buffer->setElementSize(sizeof(float));
    energy_buffer->setSize(grid.energy.size());
    std::vector<float> floats2(grid.energy.size());
    std::transform(std::begin(grid.energy), std::end(grid.energy), std::begin(floats2), [&](const double& value) { return static_cast<float>(value); });
    memcpy(energy_buffer->map(), floats2.data(), grid.energy.size() * sizeof(float));
    energy_buffer->unmap();
    // printf("EnergyGrid_.energy buffer id: %d\n", energy_buffer->getId());

    Nuclide_::EnergyGrid_ grid_(grid, grid_index_buffer->getId(), energy_buffer->getId());
    grids.push_back(grid_);
  }
  Buffer grid_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
  grid_buffer->setElementSize(sizeof(Nuclide_::EnergyGrid_));
  grid_buffer->setSize(nuclide->grid_.size());
  memcpy(grid_buffer->map(), grids.data(), grids.size() * sizeof(Nuclide_::EnergyGrid_));
  grid_buffer->unmap();
  // printf("nuclide_.grid_ buffer id: %d\n", grid_buffer->getId());

  // printf("nuclide->grid_[0].grid_index.size(): %lu\n", nuclide->grid_[0].grid_index.size());
  // printf("nuclide->grid_[0].grid_index[0]: %i\n", nuclide->grid_[0].grid_index[0]);
  // printf("nuclide->grid_[0].grid_index[-1]: %i\n", nuclide->grid_[0].grid_index[nuclide->grid_[0].grid_index.size() - 1]);
  // printf("nuclide->grid_[0].energy[76399]: %lf\n", nuclide->grid_[0].energy[76399]);
  // printf("nuclide->grid_[0].energy[76400]: %lf\n", nuclide->grid_[0].energy[76400]);

  // Nuclide cross sections
  std::vector<rtBufferId<float, 1>> xss;
  for (auto& xs : nuclide->xs_) {
    // printf("xs.size(): %d\n", xs.size());

    Buffer xs_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
    xs_buffer->setElementSize(sizeof(float));
    xs_buffer->setSize(xs.size());
    std::vector<float> floats3(xs.size());
    std::transform(std::begin(xs), std::end(xs), std::begin(floats3), [&](const double& value) { return static_cast<float>(value); });
    memcpy(xs_buffer->map(), floats3.data(), xs.size() * sizeof(float));
    xs_buffer->unmap();
    // printf("xs_.xs_ buffer id: %d\n", xs_buffer->getId());

    xss.push_back(xs_buffer->getId());
  }
  // printf("xss.size(): %d\n", xss.size());

  Buffer xs_buffers = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_BUFFER_ID);
  xs_buffers->setSize(xss.size());
  memcpy(xs_buffers->map(), xss.data(), xss.size() * sizeof(rtBufferId<rtBufferId<float, 1>, 1>));
  xs_buffers->unmap();
  // printf("Nuclide.xs_ buffer id: %d\n", xs_buffers->getId());

  // printf("nuclide->xs_[0].size(): %d\n", nuclide->xs_[0].size());
  // printf("nuclide->xs_[0](0,1,2,3,4,5, 0): %lf %lf %lf %lf %lf %lf\n",
  //        nuclide->xs_[0](0, 0), nuclide->xs_[0](1, 0), nuclide->xs_[0](2, 0),
  //        nuclide->xs_[0](3, 0), nuclide->xs_[0](4, 0), nuclide->xs_[0](5, 0));
  // printf("nuclide->xs_[0](0, 0,1,2,3,4,5): %lf %lf %lf %lf %lf %lf\n",
  //        nuclide->xs_[0](0, 0), nuclide->xs_[0](0, 1), nuclide->xs_[0](0, 2),
  //        nuclide->xs_[0](0, 3), nuclide->xs_[0](0, 4), nuclide->xs_[0](0, 5));
  // printf("nuclide->xs_[0](76512, 0): %lf\n", nuclide->xs_[0](76512, 0));
  // printf("nuclide->xs_[0](0, 76512): %lf\n", nuclide->xs_[0](0, 76512));

  // Nuclide.fission_rx_
  Buffer fission_rx_buffer = initialize_nuclide_reactions(context, nuclide->fission_rx_);

  // Nuclide.reactions_
  Buffer reactions_buffer = initialize_nuclide_reactions(context, nuclide->reactions_);

  // Nuclide index_inelastic_scatter
  Buffer index_inelastic_scatter_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
  index_inelastic_scatter_buffer->setElementSize(sizeof(int));
  index_inelastic_scatter_buffer->setSize(nuclide->index_inelastic_scatter_.size());
  memcpy(index_inelastic_scatter_buffer->map(), nuclide->index_inelastic_scatter_.data(),
         nuclide->index_inelastic_scatter_.size() * sizeof(int));
  index_inelastic_scatter_buffer->unmap();
  // printf("nuclide_.index_inelastic_scatter_ buffer id: %d\n", index_inelastic_scatter_buffer->getId());

  // // Angle distribution energy buffer
  // Buffer angle_distribution_energy_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
  // angle_distribution_energy_buffer->setElementSize(sizeof(double));
  // auto *uae = (UncorrelatedAngleEnergy *) nuclide->fission_rx_[0]->products_[0].distribution_[0].get();
  // auto &ad = uae->angle_;
  // angle_distribution_energy_buffer->setSize(ad.energy_.size());
  // memcpy(angle_distribution_energy_buffer->map(), ad.energy_.data(), ad.energy_.size() * sizeof(double));
  // angle_distribution_energy_buffer->unmap();


  Nuclide_ nuclide_ = Nuclide_(nuclide,
    kTs_buffer->getId(),
    grid_buffer->getId(),
    xs_buffers->getId(),
    fission_rx_buffer->getId(),
    reactions_buffer->getId(),
    index_inelastic_scatter_buffer->getId());
  context["nuclide"]->setUserData(sizeof(nuclide_), &nuclide_);


  // ===========================================================================
  // Input buffers
  // ===========================================================================

  // Random seed buffers
  Buffer prn_seed_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
  prn_seed_buffer->setElementSize(sizeof(uint64_t));
  prn_seed_buffer->setSize(simulation::work_per_rank * N_STREAMS);
  context["prn_seed_buffer"]->set(prn_seed_buffer);

  Buffer stream_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT);
  stream_buffer->setSize(simulation::work_per_rank);
  context["stream_buffer"]->set(stream_buffer);

  printf("Done initialising device data\n");
}

Buffer initialize_nuclide_reactions(Context context, std::vector<Reaction*> reactions) {
  std::vector<Reaction_> reactions_;
  for (auto &reaction : reactions) {

    // Reaction.products_
    std::vector<ReactionProduct_> products;
    for (auto &product : reaction->products_) {
      ReactionProduct_ product_;

      // ReactionProduct.yield_
      bool is_polynomial_yield = true;
      Polynomial_ polynomial_yield;
      Tabulated1D_ tabulated_1d_yield;
      Polynomial *polynomial = dynamic_cast<Polynomial *>(product.yield_.get());
      if (polynomial) {
        initialize_polynomial(polynomial_yield, context, *polynomial);
      } else {
        is_polynomial_yield = false;
        Tabulated1D *tabulated_1d = dynamic_cast<Tabulated1D *>(product.yield_.get());
        if (tabulated_1d) {
         initialize_tabulated_1d(tabulated_1d_yield, context, *tabulated_1d);
        } else {
          throw "Could not determine yield type";
        }
      }

      // ReactionProduct.applicability_
      std::vector<Tabulated1D_> applicabilities;
      for (auto &applicability : product.applicability_) {
        Tabulated1D_ applicability_;
        initialize_tabulated_1d(applicability_, context, applicability);
        applicabilities.push_back(applicability_);
      }
      Buffer applicability_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
      applicability_buffer->setElementSize(sizeof(Tabulated1D_));
      applicability_buffer->setSize(product.applicability_.size());
      memcpy(applicability_buffer->map(), applicabilities.data(),
             applicabilities.size() * sizeof(Tabulated1D_));
      applicability_buffer->unmap();
      // printf("Reaction.applicability buffer id: %d\n", applicability_buffer->getId());

      // ReactionProduct.distribution_
      // FIXME: All products have only one distribution, so we can assume for
      //  now that the distribution vector will contain only a single type.
      //  This is probably not true in general though and will most likely need
      //  to be fixed.
      Buffer distribution_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
      // printf("Reaction.distribution buffer id: %d\n", distribution_buffer->getId());

      if (UncorrelatedAngleEnergy *uae = dynamic_cast<UncorrelatedAngleEnergy *>(product.distribution_[0].get())) {
        product_.distribution_type = AngleEnergy_::Type::uncorrelated;

        UncorrelatedAngleEnergy_ energy_;
        energy_.fission_ = uae->fission_;

        AngleDistribution_ angle_;
        initialize_angle_distribution(angle_, context, uae->angle_);
        energy_.angle_ = angle_;
        energy_.angle_empty = uae->angle_.empty();

        if (ContinuousTabular *ct = dynamic_cast<ContinuousTabular *>(uae->energy_.get())) {
          // printf("ContinuousTabular\n");
          energy_.energy_type = EnergyDistribution_::Type::continuous;
          ContinuousTabular_ ct_;
          initialize_continuous_tabular(ct_, context, ct);
          energy_.energy_ct = ct_;
        }
        else if (DiscretePhoton *dp = dynamic_cast<DiscretePhoton *>(uae->energy_.get())) {
          // printf("DiscretePhoton\n");
          energy_.energy_type = EnergyDistribution_::Type::discrete_photon;
          DiscretePhoton_ dp_;
          initialize_discrete_photon(dp_, context, dp);
          // energy_.energy_ = dp_;
        }
        else if (LevelInelastic *li = dynamic_cast<LevelInelastic *>(uae->energy_.get())) {
          // printf("LevelInelastic\n");
          energy_.energy_type = EnergyDistribution_::Type::level;
          LevelInelastic_ li_;
          initialize_level_inelastic(li_, context, li);
          energy_.energy_li = li_;
        }
        else {
          // printf("None\n");
          energy_.energy_type = EnergyDistribution_::Type::none;
          // EnergyDistribution_ e_;
          // energy_.energy_ = e_;
        }

        distribution_buffer->setElementSize(sizeof(UncorrelatedAngleEnergy_));
        distribution_buffer->setSize(product.distribution_.size()); // FIXME: support more than one
        memcpy(distribution_buffer->map(), &energy_, product.distribution_.size() * sizeof(UncorrelatedAngleEnergy_));
        distribution_buffer->unmap();
        product_.distribution_uae = distribution_buffer->getId();
      }
      else if (KalbachMann *km = dynamic_cast<KalbachMann *>(product.distribution_[0].get())) {
        product_.distribution_type = AngleEnergy_::Type::kalbach_mann;

        KalbachMann_ km_;
        initialize_kalbach_mann(km_, context, km);

        distribution_buffer->setElementSize(sizeof(KalbachMann_));
        distribution_buffer->setSize(product.distribution_.size()); // FIXME: support more than one
        memcpy(distribution_buffer->map(), &km_, product.distribution_.size() * sizeof(KalbachMann_));
        distribution_buffer->unmap();
        product_.distribution_km = distribution_buffer->getId();

      }
      else {
        throw "Unsupported AngleEnergy";
      }

      product_.particle_ = product.particle_;
      product_.emission_mode_ = product.emission_mode_;
      product_.decay_rate_ = product.decay_rate_;
      product_.is_polynomial_yield = is_polynomial_yield;
      product_.polynomial_yield_ = polynomial_yield;
      product_.tabulated_1d_yield_ = tabulated_1d_yield;
      product_.applicability_ = applicability_buffer->getId();

      products.push_back(product_);
    }
    Buffer products_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
    products_buffer->setElementSize(sizeof(ReactionProduct_));
    products_buffer->setSize(reaction->products_.size());
    memcpy(products_buffer->map(), products.data(), products.size() * sizeof(ReactionProduct_));
    products_buffer->unmap();
    // printf("Reaction.products buffer id: %d\n", products_buffer->getId());

    // Reaction.xs_
    std::vector<Reaction_::TemperatureXS_> xss;
    for (auto &xs : reaction->xs_) {
      Buffer value_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
      value_buffer->setElementSize(sizeof(float));
      value_buffer->setSize(xs.value.size());
      std::vector<float> floats(xs.value.size());
      std::transform(std::begin(xs.value), std::end(xs.value), std::begin(floats), [&](const double& value) { return static_cast<float>(value); });
      memcpy(value_buffer->map(), floats.data(), xs.value.size() * sizeof(float));
      value_buffer->unmap();

      Reaction_::TemperatureXS_ xs_(&xs, value_buffer->getId());
      xss.push_back(xs_);

      // printf("TemperatureXS_.value_ buffer id: %d\n", xs_.value_.getId());
    }
    Buffer temperature_xs_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
    temperature_xs_buffer->setElementSize(sizeof(Reaction_::TemperatureXS_));
    temperature_xs_buffer->setSize(reaction->xs_.size());
    memcpy(temperature_xs_buffer->map(), xss.data(), xss.size() * sizeof(Reaction_::TemperatureXS_));
    temperature_xs_buffer->unmap();
    // printf("Reaction.temperature_xs buffer id: %d\n", temperature_xs_buffer->getId());

    Reaction_ reaction_(reaction, temperature_xs_buffer->getId(), products_buffer->getId());
    reactions_.push_back(reaction_);
  }
  Buffer reactions_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
  reactions_buffer->setElementSize(sizeof(Reaction_));
  reactions_buffer->setSize(reactions.size());
  memcpy(reactions_buffer->map(), reactions_.data(), reactions_.size() * sizeof(Reaction_));
  reactions_buffer->unmap();
  // printf("Reactions buffer id: %d\n", reactions_buffer->getId());

  return reactions_buffer;
}

void initialize_tabulated_1d(Tabulated1D_& tabulated_1d_, Context context, const Tabulated1D& tabulated_1d) {
  // Tabulated1D.nbt_
  Buffer nbt_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
  nbt_buffer->setElementSize(sizeof(int));
  nbt_buffer->setSize(tabulated_1d.nbt_.size());
  memcpy(nbt_buffer->map(), tabulated_1d.nbt_.data(),
         tabulated_1d.nbt_.size() * sizeof(int));
  nbt_buffer->unmap();

  // Tabulated1D.int_
  Buffer int_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
  int_buffer->setElementSize(sizeof(Interpolation));
  int_buffer->setSize(tabulated_1d.int_.size());
  memcpy(int_buffer->map(), tabulated_1d.int_.data(),
         tabulated_1d.int_.size() * sizeof(Interpolation));
  int_buffer->unmap();

  // Tabulated1D.x_
  Buffer x_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
  x_buffer->setElementSize(sizeof(float));
  x_buffer->setSize(tabulated_1d.x_.size());
  std::vector<float> floats(tabulated_1d.x_.size());
  std::transform(std::begin(tabulated_1d.x_), std::end(tabulated_1d.x_), std::begin(floats), [&](const double& value) { return static_cast<float>(value); });
  memcpy(x_buffer->map(), floats.data(), tabulated_1d.x_.size() * sizeof(float));
  x_buffer->unmap();

  // Tabulated1D.y_
  Buffer y_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
  y_buffer->setElementSize(sizeof(float));
  y_buffer->setSize(tabulated_1d.y_.size());
  std::vector<float> floats2(tabulated_1d.y_.size());
  std::transform(std::begin(tabulated_1d.y_), std::end(tabulated_1d.y_), std::begin(floats2), [&](const double& value) { return static_cast<float>(value); });
  memcpy(y_buffer->map(), floats2.data(), tabulated_1d.y_.size() * sizeof(float));
  y_buffer->unmap();

  tabulated_1d_.n_regions_ = tabulated_1d.n_regions_;
  tabulated_1d_.n_pairs_ = tabulated_1d.n_pairs_;
  tabulated_1d_.nbt_ = nbt_buffer->getId();
  tabulated_1d_.int_ = int_buffer->getId();
  tabulated_1d_.x_ =  x_buffer->getId();
  tabulated_1d_.y_ = y_buffer->getId();

  // printf("Tabulated1D.nbt_ buffer id: %d\n", tabulated_1d_.nbt_.getId());
  // printf("Tabulated1D.int_ buffer id: %d\n", tabulated_1d_.int_.getId());
  // printf("Tabulated1D.x_ buffer id: %d\n", tabulated_1d_.x_.getId());
  // printf("Tabulated1D.y_ buffer id: %d\n", tabulated_1d_.y_.getId());
};

void initialize_polynomial(Polynomial_& polynomial_, Context context, const Polynomial& polynomial) {
  Buffer yield_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
  yield_buffer->setElementSize(sizeof(float));
  yield_buffer->setSize(polynomial.coef_.size());
  std::vector<float> floats(polynomial.coef_.size());
  std::transform(std::begin(polynomial.coef_), std::end(polynomial.coef_), std::begin(floats), [&](const double& value) { return static_cast<float>(value); });
  memcpy(yield_buffer->map(), floats.data(), polynomial.coef_.size() * sizeof(int));
  yield_buffer->unmap();
  // printf("Polynomial.yield buffer id: %d\n", yield_buffer->getId());

  polynomial_.coef_ = yield_buffer->getId();
  polynomial_.num_coeffs = polynomial.coef_.size();
}

void initialize_angle_distribution(AngleDistribution_& angle_, Context context, AngleDistribution& angle) {
  // AngleDistribution.energy_
  Buffer energy_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
  energy_buffer->setElementSize(sizeof(float));
  energy_buffer->setSize(angle.energy_.size());
  std::vector<float> floats(angle.energy_.size());
  std::transform(std::begin(angle.energy_), std::end(angle.energy_), std::begin(floats), [&](const double& value) { return static_cast<float>(value); });
  memcpy(energy_buffer->map(), floats.data(), angle.energy_.size() * sizeof(float));
  energy_buffer->unmap();
  // printf("AngleDistribution.energy_ buffer id: %d\n", energy_buffer->getId());

  // AngleDistribution.distribution_
  std::vector<Tabular_> distributions;
  for (auto &distribution : angle.distribution_) {
    Tabular *tabular = (Tabular*) distribution.get();

    // Tabular.x_
    Buffer x_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
    x_buffer->setElementSize(sizeof(float));
    x_buffer->setSize(tabular->x_.size());
    std::vector<float> floats2(tabular->x_.size());
    std::transform(std::begin(tabular->x_), std::end(tabular->x_), std::begin(floats2), [&](const double& value) { return static_cast<float>(value); });
    memcpy(x_buffer->map(), floats2.data(), tabular->x_.size() * sizeof(float));
    x_buffer->unmap();

    // Tabular.p_
    Buffer p_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
    p_buffer->setElementSize(sizeof(float));
    p_buffer->setSize(tabular->p_.size());
    std::vector<float> floats3(tabular->p_.size());
    std::transform(std::begin(tabular->p_), std::end(tabular->p_), std::begin(floats3), [&](const double& value) { return static_cast<float>(value); });
    memcpy(p_buffer->map(), floats3.data(), tabular->p_.size() * sizeof(float));
    p_buffer->unmap();

    // Tabular.c_
    Buffer c_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
    c_buffer->setElementSize(sizeof(float));
    c_buffer->setSize(tabular->c_.size());
    std::vector<float> floats4(tabular->c_.size());
    std::transform(std::begin(tabular->c_), std::end(tabular->c_), std::begin(floats4), [&](const double& value) { return static_cast<float>(value); });
    memcpy(c_buffer->map(), floats4.data(), tabular->c_.size() * sizeof(float));
    c_buffer->unmap();

    Tabular_ tabular_(x_buffer->getId(), p_buffer->getId(), c_buffer->getId(), tabular->interp_);
    distributions.push_back(tabular_);

    // printf("angle_.distribution_.x_ buffer id: %d\n", tabular_.x_.getId());
    // printf("angle_.distribution_.p_ buffer id: %d\n", tabular_.p_.getId());
    // printf("angle_.distribution_.c_ buffer id: %d\n", tabular_.c_.getId());
  }

  Buffer distribution_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
  distribution_buffer->setElementSize(sizeof(Tabular_));
  distribution_buffer->setSize(distributions.size());
  memcpy(distribution_buffer->map(), distributions.data(), distributions.size() * sizeof(Tabular_));
  distribution_buffer->unmap();
  // printf("AngleDistribution.distribution buffer id: %d\n", distribution_buffer->getId());

  angle_.energy_ = energy_buffer->getId();
  angle_.distribution_ = distribution_buffer->getId();
}

void initialize_continuous_tabular(ContinuousTabular_& energy_, Context context, ContinuousTabular* ct) {
  // ContinuousTabular.breakpoints_
  Buffer breakpoints_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
  breakpoints_buffer->setElementSize(sizeof(int));
  breakpoints_buffer->setSize(ct->breakpoints_.size());
  memcpy(breakpoints_buffer->map(), ct->breakpoints_.data(), ct->breakpoints_.size() * sizeof(int));
  breakpoints_buffer->unmap();

  // ContinuousTabular.interpolation_
  Buffer interpolation_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
  interpolation_buffer->setElementSize(sizeof(Interpolation));
  interpolation_buffer->setSize(ct->interpolation_.size());
  memcpy(interpolation_buffer->map(), ct->interpolation_.data(), ct->interpolation_.size() * sizeof(Interpolation));
  interpolation_buffer->unmap();

  // ContinuousTabular.energy_
  Buffer energy_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
  energy_buffer->setElementSize(sizeof(float));
  energy_buffer->setSize(ct->energy_.size());
  std::vector<float> floats(ct->energy_.size());
  std::transform(std::begin(ct->energy_), std::end(ct->energy_), std::begin(floats), [&](const double& value) { return static_cast<float>(value); });
  memcpy(energy_buffer->map(), floats.data(), ct->energy_.size() * sizeof(float));
  energy_buffer->unmap();

  // ContinuousTabular.distribution_
  std::vector<ContinuousTabular_::CTTable_> distributions;
  for (auto &distribution : ct->distribution_) {

    // CTTable.e_out
    Buffer e_out_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
    e_out_buffer->setElementSize(sizeof(float));
    e_out_buffer->setSize(distribution.e_out.size());
    std::vector<float> floats2(distribution.e_out.size());
    std::transform(std::begin(distribution.e_out), std::end(distribution.e_out), std::begin(floats2), [&](const double& value) { return static_cast<float>(value); });
    memcpy(e_out_buffer->map(), floats2.data(), distribution.e_out.size() * sizeof(float));
    e_out_buffer->unmap();

    // CTTable.p
    Buffer p_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
    p_buffer->setElementSize(sizeof(float));
    p_buffer->setSize(distribution.p.size());
    std::vector<float> floats3(distribution.p.size());
    std::transform(std::begin(distribution.p), std::end(distribution.p), std::begin(floats3), [&](const double& value) { return static_cast<float>(value); });
    memcpy(p_buffer->map(), floats3.data(), distribution.p.size() * sizeof(float));
    p_buffer->unmap();

    // CTTable.c
    Buffer c_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
    c_buffer->setElementSize(sizeof(float));
    c_buffer->setSize(distribution.c.size());
    std::vector<float> floats4(distribution.c.size());
    std::transform(std::begin(distribution.c), std::end(distribution.c), std::begin(floats4), [&](const double& value) { return static_cast<float>(value); });
    memcpy(c_buffer->map(), floats4.data(), distribution.c.size() * sizeof(float));
    c_buffer->unmap();

    ContinuousTabular_::CTTable_ distribution_;
    distribution_.interpolation = distribution.interpolation;
    distribution_.n_discrete = distribution.n_discrete;
    distribution_.e_out = e_out_buffer->getId();
    distribution_.p = p_buffer->getId();
    distribution_.c = c_buffer->getId();

    distributions.push_back(distribution_);

    // printf("energy_.distribution_.e_out buffer id: %d\n", distribution_.e_out.getId());
    // printf("energy_.distribution_.p buffer id: %d\n", distribution_.p.getId());
    // printf("energy_.distribution_.c buffer id: %d\n", distribution_.c.getId());
  }
  Buffer distribution_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
  distribution_buffer->setElementSize(sizeof(ContinuousTabular_::CTTable_));
  distribution_buffer->setSize(distributions.size());
  memcpy(distribution_buffer->map(), distributions.data(), distributions.size() * sizeof(ContinuousTabular_::CTTable_));
  distribution_buffer->unmap();

  energy_.n_region_ = ct->n_region_;
  energy_.breakpoints_ = breakpoints_buffer->getId();
  energy_.interpolation_ = interpolation_buffer->getId();
  energy_.energy_ = energy_buffer->getId();
  energy_.distribution_ = distribution_buffer->getId();

  // printf("energy_.breakpoints_ buffer id: %d\n", energy_.breakpoints_.getId());
  // printf("energy_.interpolation_ buffer id: %d\n", energy_.interpolation_.getId());
  // printf("energy_.energy_ buffer id: %d\n", energy_.energy_.getId());
  // printf("energy_.distribution_ buffer id: %d\n", energy_.distribution_.getId());
}

void initialize_discrete_photon(DiscretePhoton_ &energy_, Context context, DiscretePhoton *dp) {
  energy_.primary_flag_ = dp->primary_flag_;
  energy_.A_ = static_cast<float>(dp->A_);
  energy_.energy_ = static_cast<float>(dp->energy_);
}

void initialize_level_inelastic(LevelInelastic_ &energy_, Context context, LevelInelastic *li) {
  energy_.mass_ratio_ = static_cast<float>(li->mass_ratio_);
  energy_.threshold_ = static_cast<float>(li->threshold_);
}

void initialize_kalbach_mann(KalbachMann_ &energy_, Context context, KalbachMann *km) {
  energy_.n_region_ = km->n_region_;

  // KalbachMann.breakpoints_
  Buffer breakpoints_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
  breakpoints_buffer->setElementSize(sizeof(int));
  breakpoints_buffer->setSize(km->breakpoints_.size());
  memcpy(breakpoints_buffer->map(), km->breakpoints_.data(), km->breakpoints_.size() * sizeof(int));
  breakpoints_buffer->unmap();

  // KalbachMann.interpolation_
  Buffer interpolation_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
  interpolation_buffer->setElementSize(sizeof(Interpolation));
  interpolation_buffer->setSize(km->interpolation_.size());
  memcpy(interpolation_buffer->map(), km->interpolation_.data(), km->interpolation_.size() * sizeof(Interpolation));
  interpolation_buffer->unmap();

  // KalbachMann.energy_
  Buffer energy_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
  energy_buffer->setElementSize(sizeof(float));
  energy_buffer->setSize(km->energy_.size());
  std::vector<float> floats(km->energy_.size());
  std::transform(std::begin(km->energy_), std::end(km->energy_), std::begin(floats), [&](const double& value) { return static_cast<float>(value); });
  memcpy(energy_buffer->map(), floats.data(), km->energy_.size() * sizeof(float));
  energy_buffer->unmap();

  // KalbachMann.distribution_
  std::vector<KalbachMann_::KMTable_> distributions;
  for (auto &distribution : km->distribution_) {
    
    // KMTable.e_out
    Buffer e_out_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
    e_out_buffer->setElementSize(sizeof(float));
    e_out_buffer->setSize(distribution.e_out.size());
    std::vector<float> floats2(distribution.e_out.size());
    std::transform(std::begin(distribution.e_out), std::end(distribution.e_out), std::begin(floats2), [&](const double& value) { return static_cast<float>(value); });
    memcpy(e_out_buffer->map(), floats2.data(), distribution.e_out.size() * sizeof(float));
    e_out_buffer->unmap();

    // KMTable.p
    Buffer p_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
    p_buffer->setElementSize(sizeof(float));
    p_buffer->setSize(distribution.p.size());
    std::vector<float> floats3(distribution.p.size());
    std::transform(std::begin(distribution.p), std::end(distribution.p), std::begin(floats3), [&](const double& value) { return static_cast<float>(value); });
    memcpy(p_buffer->map(), floats3.data(), distribution.p.size() * sizeof(float));
    p_buffer->unmap();

    // KMTable.c
    Buffer c_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
    c_buffer->setElementSize(sizeof(float));
    c_buffer->setSize(distribution.c.size());
    std::vector<float> floats4(distribution.c.size());
    std::transform(std::begin(distribution.c), std::end(distribution.c), std::begin(floats4), [&](const double& value) { return static_cast<float>(value); });
    memcpy(c_buffer->map(), floats4.data(), distribution.c.size() * sizeof(float));
    c_buffer->unmap();

    // KMTable.r
    Buffer r_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
    r_buffer->setElementSize(sizeof(float));
    r_buffer->setSize(distribution.r.size());
    std::vector<float> floats5(distribution.r.size());
    std::transform(std::begin(distribution.r), std::end(distribution.r), std::begin(floats5), [&](const double& value) { return static_cast<float>(value); });
    memcpy(r_buffer->map(), floats5.data(), distribution.r.size() * sizeof(float));
    r_buffer->unmap();

    // KMTable.a
    Buffer a_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
    a_buffer->setElementSize(sizeof(float));
    a_buffer->setSize(distribution.a.size());
    std::vector<float> floats6(distribution.a.size());
    std::transform(std::begin(distribution.a), std::end(distribution.a), std::begin(floats6), [&](const double& value) { return static_cast<float>(value); });
    memcpy(a_buffer->map(), floats6.data(), distribution.a.size() * sizeof(float));
    a_buffer->unmap();

    KalbachMann_::KMTable_ distribution_;
    distribution_.n_discrete = distribution.n_discrete;
    distribution_.interpolation = distribution.interpolation;
    distribution_.e_out = e_out_buffer->getId();
    distribution_.p = p_buffer->getId();
    distribution_.c = c_buffer->getId();
    distribution_.r = r_buffer->getId();
    distribution_.a = a_buffer->getId();

    distributions.push_back(distribution_);
  }
  Buffer distribution_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
  distribution_buffer->setElementSize(sizeof(KalbachMann_::KMTable_));
  distribution_buffer->setSize(distributions.size());
  memcpy(distribution_buffer->map(), distributions.data(), distributions.size() * sizeof(KalbachMann_::KMTable_));
  distribution_buffer->unmap();

  energy_.breakpoints_ = breakpoints_buffer->getId();
  energy_.interpolation_ = interpolation_buffer->getId();
  energy_.energy_ = energy_buffer->getId();
  energy_.distribution_ = distribution_buffer->getId();
}
