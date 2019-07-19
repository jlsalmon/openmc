#ifndef OPENMC_PARTICLE_H
#define OPENMC_PARTICLE_H

//! \file particle.h
//! \brief Particle type

#include <array>
#include <cstdint>
#include <memory> // for unique_ptr
#include <sstream>
#include <string>

#include "openmc/constants.h"
#include "openmc/position.h"

#include <optix_world.h>

namespace openmc {

//==============================================================================
// Constants
//==============================================================================

// Since cross section libraries come with different numbers of delayed groups
// (e.g. ENDF/B-VII.1 has 6 and JEFF 3.1.1 has 8 delayed groups) and we don't
// yet know what cross section library is being used when the tallies.xml file
// is read in, we want to have an upper bound on the size of the array we
// use to store the bins for delayed group tallies.
constexpr int MAX_DELAYED_GROUPS {8};

// Maximum number of lost particles
constexpr int MAX_LOST_PARTICLES {10};

// Maximum number of lost particles, relative to the total number of particles
constexpr double REL_MAX_LOST_PARTICLES {1.0e-6};

constexpr double CACHE_INVALID {-1.0};

//==============================================================================
// Class declarations
//==============================================================================

struct LocalCoord {
  Position r; //!< particle position
  Direction u; //!< particle direction
  int cell {-1};
  int universe {-1};
  int lattice {-1};
  int lattice_x {-1};
  int lattice_y {-1};
  int lattice_z {-1};
  bool rotated {false};  //!< Is the level rotated?

  //! clear data from a single coordinate level
#ifndef __CUDA_ARCH__
  void reset();
#else
  __forceinline__ __device__ void reset()
  {
    cell = C_NONE;
    universe = C_NONE;
    lattice = C_NONE;
    lattice_x = 0;
    lattice_y = 0;
    rotated = false;
  }
#endif
};

struct LocalCoord_ {
  Position_ r; //!< particle position
  Direction_ u; //!< particle direction
  int cell {-1};
  int universe {-1};
  int lattice {-1};
  int lattice_x {-1};
  int lattice_y {-1};
  int lattice_z {-1};
  bool rotated {false};  //!< Is the level rotated?

  //! clear data from a single coordinate level
  __forceinline__ __device__ void reset()
  {
    cell = C_NONE;
    universe = C_NONE;
    lattice = C_NONE;
    lattice_x = 0;
    lattice_y = 0;
    rotated = false;
  }
};

//==============================================================================
//! Cached microscopic cross sections for a particular nuclide at the current
//! energy
//==============================================================================

struct NuclideMicroXS {
  // Microscopic cross sections in barns
  double total;            //!< total cross section
  double absorption;       //!< absorption (disappearance)
  double fission;          //!< fission
  double nu_fission;       //!< neutron production from fission

  double elastic;          //!< If sab_frac is not 1 or 0, then this value is
                           //!<   averaged over bound and non-bound nuclei
  double thermal;          //!< Bound thermal elastic & inelastic scattering
  double thermal_elastic;  //!< Bound thermal elastic scattering
  double photon_prod;      //!< microscopic photon production xs

  // Cross sections for depletion reactions (note that these are not stored in
  // macroscopic cache)
  double reaction[DEPLETION_RX.size()];

  // Indicies and factors needed to compute cross sections from the data tables
  int index_grid;        //!< Index on nuclide energy grid
  int index_temp;        //!< Temperature index for nuclide
  double interp_factor;  //!< Interpolation factor on nuc. energy grid
  int index_sab {-1};    //!< Index in sab_tables
  int index_temp_sab;    //!< Temperature index for sab_tables
  double sab_frac;       //!< Fraction of atoms affected by S(a,b)
  bool use_ptable;       //!< In URR range with probability tables?

  // Energy and temperature last used to evaluate these cross sections.  If
  // these values have changed, then the cross sections must be re-evaluated.
  double last_E {0.0};      //!< Last evaluated energy
  double last_sqrtkT {0.0}; //!< Last temperature in sqrt(Boltzmann constant
                            //!< * temperature (eV))
};

struct NuclideMicroXS_ {
  // Microscopic cross sections in barns
  float total;            //!< total cross section
  float absorption;       //!< absorption (disappearance)
  float fission;          //!< fission
  float nu_fission;       //!< neutron production from fission

  float elastic;          //!< If sab_frac is not 1 or 0, then this value is
  //!<   averaged over bound and non-bound nuclei
  float thermal;          //!< Bound thermal elastic & inelastic scattering
  float thermal_elastic;  //!< Bound thermal elastic scattering
  float photon_prod;      //!< microscopic photon production xs

  // Cross sections for depletion reactions (note that these are not stored in
  // macroscopic cache)
  float reaction[DEPLETION_RX.size()];

  // Indicies and factors needed to compute cross sections from the data tables
  int index_grid;        //!< Index on nuclide energy grid
  int index_temp;        //!< Temperature index for nuclide
  float interp_factor;  //!< Interpolation factor on nuc. energy grid
  int index_sab {-1};    //!< Index in sab_tables
  int index_temp_sab;    //!< Temperature index for sab_tables
  float sab_frac;       //!< Fraction of atoms affected by S(a,b)
  bool use_ptable;       //!< In URR range with probability tables?

  // Energy and temperature last used to evaluate these cross sections.  If
  // these values have changed, then the cross sections must be re-evaluated.
  float last_E {0.0};      //!< Last evaluated energy
  float last_sqrtkT {0.0}; //!< Last temperature in sqrt(Boltzmann constant
  //!< * temperature (eV))
};

//==============================================================================
//! Cached microscopic photon cross sections for a particular element at the
//! current energy
//==============================================================================

struct ElementMicroXS {
  int index_grid; //!< index on element energy grid
  double last_E {0.0}; //!< last evaluated energy in [eV]
  double interp_factor; //!< interpolation factor on energy grid
  double total; //!< microscopic total photon xs
  double coherent; //!< microscopic coherent xs
  double incoherent; //!< microscopic incoherent xs
  double photoelectric; //!< microscopic photoelectric xs
  double pair_production; //!< microscopic pair production xs
};

struct ElementMicroXS_ {
  int index_grid; //!< index on element energy grid
  float last_E {0.0}; //!< last evaluated energy in [eV]
  float interp_factor; //!< interpolation factor on energy grid
  float total; //!< microscopic total photon xs
  float coherent; //!< microscopic coherent xs
  float incoherent; //!< microscopic incoherent xs
  float photoelectric; //!< microscopic photoelectric xs
  float pair_production; //!< microscopic pair production xs
};

//==============================================================================
// MACROXS contains cached macroscopic cross sections for the material a
// particle is traveling through
//==============================================================================

struct MacroXS {
  double total;         //!< macroscopic total xs
  double absorption;    //!< macroscopic absorption xs
  double fission;       //!< macroscopic fission xs
  double nu_fission;    //!< macroscopic production xs
  double photon_prod;   //!< macroscopic photon production xs

  // Photon cross sections
  double coherent;        //!< macroscopic coherent xs
  double incoherent;      //!< macroscopic incoherent xs
  double photoelectric;   //!< macroscopic photoelectric xs
  double pair_production; //!< macroscopic pair production xs
};

struct MacroXS_ {
  float total;         //!< macroscopic total xs
  float absorption;    //!< macroscopic absorption xs
  float fission;       //!< macroscopic fission xs
  float nu_fission;    //!< macroscopic production xs
  float photon_prod;   //!< macroscopic photon production xs

  // Photon cross sections
  float coherent;        //!< macroscopic coherent xs
  float incoherent;      //!< macroscopic incoherent xs
  float photoelectric;   //!< macroscopic photoelectric xs
  double pair_production; //!< macroscopic pair production xs
};

//============================================================================
//! State of a particle being transported through geometry
//============================================================================

class Particle {
public:
  //==========================================================================
  // Aliases and type definitions

  //! Particle types
  enum class Type {
    neutron, photon, electron, positron
  };

  //! Saved ("banked") state of a particle
  struct Bank {
    Position r;
    Direction u;
    double E;
    double wgt;
    int delayed_group;
    Type particle;
  };

  //==========================================================================
  // Constructors

  Particle();

  //==========================================================================
  // Methods and accessors

  // Accessors for position in global coordinates
  Position& r() { return coord_[0].r; }
  const Position& r() const { return coord_[0].r; }

  // Accessors for position in local coordinates
  Position& r_local() { return coord_[n_coord_ - 1].r; }
  const Position& r_local() const { return coord_[n_coord_ - 1].r; }

  // Accessors for direction in global coordinates
  Direction& u() { return coord_[0].u; }
  const Direction& u() const { return coord_[0].u; }

  // Accessors for direction in local coordinates
  Direction& u_local() { return coord_[n_coord_ - 1].u; }
  const Direction& u_local() const { return coord_[n_coord_ - 1].u; }

  //! resets all coordinate levels for the particle
  void clear();

  //! create a secondary particle
  //
  //! stores the current phase space attributes of the particle in the
  //! secondary bank and increments the number of sites in the secondary bank.
  //! \param u Direction of the secondary particle
  //! \param E Energy of the secondary particle in [eV]
  //! \param type Particle type
  void create_secondary(Direction u, double E, Type type) const;

  //! initialize from a source site
  //
  //! initializes a particle from data stored in a source site. The source
  //! site may have been produced from an external source, from fission, or
  //! simply as a secondary particle.
  //! \param src Source site data
  void from_source(const Bank* src);

  //! Transport a particle from birth to death
  void transport();

  //! Cross a surface and handle boundary conditions
  void cross_surface();

  //! mark a particle as lost and create a particle restart file
  //! \param message A warning message to display
  void mark_as_lost(const char* message);

  void mark_as_lost(const std::string& message)
  {mark_as_lost(message.c_str());}

  void mark_as_lost(const std::stringstream& message)
  {mark_as_lost(message.str());}

  //! create a particle restart HDF5 file
  void write_restart() const;

  //==========================================================================
  // Data members

  // Cross section caches
  std::vector<NuclideMicroXS> neutron_xs_; //!< Microscopic neutron cross sections
  std::vector<ElementMicroXS> photon_xs_; //!< Microscopic photon cross sections
  MacroXS macro_xs_; //!< Macroscopic cross sections

  int64_t id_;  //!< Unique ID
  Type type_ {Type::neutron};   //!< Particle type (n, p, e, etc.)

  int n_coord_ {1};              //!< number of current coordinate levels
  int cell_instance_;            //!< offset for distributed properties
  std::vector<LocalCoord> coord_; //!< coordinates for all levels

  // Particle coordinates before crossing a surface
  int n_coord_last_ {1};      //!< number of current coordinates
  std::vector<int> cell_last_;  //!< coordinates for all levels

  // Energy data
  double E_;       //!< post-collision energy in eV
  double E_last_;  //!< pre-collision energy in eV
  int g_ {0};      //!< post-collision energy group (MG only)
  int g_last_;     //!< pre-collision energy group (MG only)

  // Other physical data
  double wgt_ {1.0};     //!< particle weight
  double mu_;      //!< angle of scatter
  bool alive_ {true};     //!< is particle alive?

  // Other physical data
  Position r_last_current_; //!< coordinates of the last collision or
                            //!< reflective/periodic surface crossing for
                            //!< current tallies
  Position r_last_;   //!< previous coordinates
  Direction u_last_;  //!< previous direction coordinates
  double wgt_last_ {1.0};   //!< pre-collision particle weight
  double wgt_absorb_ {0.0}; //!< weight absorbed for survival biasing

  // What event took place
  bool fission_ {false}; //!< did particle cause implicit fission
  int event_;          //!< scatter, absorption
  int event_nuclide_;  //!< index in nuclides array
  int event_mt_;       //!< reaction MT
  int delayed_group_ {0};  //!< delayed group

  // Post-collision physical data
  int n_bank_ {0};        //!< number of fission sites banked
  double wgt_bank_ {0.0}; //!< weight of fission sites banked
  int n_delayed_bank_[MAX_DELAYED_GROUPS];  //!< number of delayed fission
                                            //!< sites banked

  // Indices for various arrays
  int surface_ {0};             //!< index for surface particle is on
  int cell_born_ {-1};      //!< index for cell particle was born in
  int material_ {-1};       //!< index for current material
  int material_last_ {-1};  //!< index for last material

  // Temperature of current cell
  double sqrtkT_ {-1.0};      //!< sqrt(k_Boltzmann * temperature) in eV
  double sqrtkT_last_ {0.0};  //!< last temperature

  // Statistical data
  int n_collision_ {0};  //!< number of collisions

  // Track output
  bool write_track_ {false};
};

struct Particle_ {

  //! Saved ("banked") state of a particle
  struct Bank_ {
    Position_ r;
    Direction_ u;
    float E;
    float wgt;
    int delayed_group;
    Particle::Type particle;
  };

  NuclideMicroXS_ neutron_xs_[1]; //!< Microscopic neutron cross sections // FIXME: support more than one
  // ElementMicroXS photon_xs_; //!< Microscopic photon cross sections
  MacroXS_ macro_xs_; //!< Macroscopic cross sections

  int64_t id_;  //!< Unique ID
  Particle::Type type_ {Particle::Type::neutron};   //!< Particle type (n, p, e, etc.)

  int n_coord_ {1};              //!< number of current coordinate levels
  int cell_instance_;            //!< offset for distributed properties

  LocalCoord_ coord_[1]; //!< coordinates for all levels

  // Particle coordinates before crossing a surface
  int n_coord_last_ {1};      //!< number of current coordinates
  int cell_last_[1];  //!< coordinates for all levels

  // Energy data
  float E_;       //!< post-collision energy in eV
  float E_last_;  //!< pre-collision energy in eV
  int g_ {0};      //!< post-collision energy group (MG only)
  int g_last_;     //!< pre-collision energy group (MG only)

  // Other physical data
  float wgt_ {1.0f};     //!< particle weight
  float mu_;      //!< angle of scatter
  bool alive_ {true};     //!< is particle alive?

  // Other physical data
  Position_ r_last_current_; //!< coordinates of the last collision or
  //!< reflective/periodic surface crossing for
  //!< current tallies
  Position_ r_last_;   //!< previous coordinates
  Direction_ u_last_;  //!< previous direction coordinates
  float wgt_last_ {1.0f};   //!< pre-collision particle weight
  float wgt_absorb_ {0.0f}; //!< weight absorbed for survival biasing

  // What event took place
  bool fission_ {false}; //!< did particle cause implicit fission
  int event_;          //!< scatter, absorption
  int event_nuclide_;  //!< index in nuclides array
  int event_mt_;       //!< reaction MT
  int delayed_group_ {0};  //!< delayed group

  // Post-collision physical data
  int n_bank_ {0};        //!< number of fission sites banked
  float wgt_bank_ {0.0f}; //!< weight of fission sites banked
  int n_delayed_bank_[MAX_DELAYED_GROUPS];  //!< number of delayed fission
  //!< sites banked

  // Indices for various arrays
  int surface_ {0};             //!< index for surface particle is on
  int cell_born_ {-1};      //!< index for cell particle was born in
  int material_ {-1};       //!< index for current material
  int material_last_ {-1};  //!< index for last material

  // Temperature of current cell
  float sqrtkT_ {-1.0f};      //!< sqrt(k_Boltzmann * temperature) in eV
  float sqrtkT_last_ {0.0f};  //!< last temperature

  // Statistical data
  int n_collision_ {0};  //!< number of collisions

  // Track output
  bool write_track_ {false};

  __host__ __forceinline__ __device__ Particle_() {
    clear();

    for (int& n : n_delayed_bank_) {
      n = 0;
    }
  };

  __forceinline__ __device__ Position_& r() { return coord_[0].r; }
  __forceinline__ __device__ const Position_& r() const { return coord_[0].r; }

  __forceinline__ __device__ Position_& r_local() { return coord_[n_coord_ - 1].r; }
  __forceinline__ __device__ const Position_& r_local() const { return coord_[n_coord_ - 1].r; }

  __forceinline__ __device__ Direction_& u() { return coord_[0].u; }
  __forceinline__ __device__ const Direction_& u() const { return coord_[0].u; }

  __forceinline__ __device__ Direction_& u_local() { return coord_[n_coord_ - 1].u; }
  __forceinline__ __device__ const Direction_& u_local() const { return coord_[n_coord_ - 1].u; }

  __forceinline__ __device__ void clear() {
    // reset any coordinate levels
    for (int i = 0; i < 1 /*FIXME: model::n_coord_levels*/; ++i) {
      coord_[i].reset();
    }
    n_coord_ = 1;
  }

  void from_source(const Particle_::Bank_& src) {
    // reset some attributes
    this->clear();
    alive_ = true;
    surface_ = 0;
    cell_born_ = C_NONE;
    material_ = C_NONE;
    n_collision_ = 0;
    fission_ = false;

    // copy attributes from source bank site
    type_ = src.particle;
    wgt_ = src.wgt;
    wgt_last_ = src.wgt;
    this->r() = src.r;
    this->u() = src.u;
    r_last_current_ = src.r;
    r_last_ = src.r;
    u_last_ = src.u;
    // if (settings::run_CE) { // FIXME: multigroup
      E_ = src.E;
      g_ = 0;
    // } else {
    //   g_ = static_cast<int>(src->E);
    //   g_last_ = static_cast<int>(src->E);
    //   E_ = data::energy_bin_avg[g_ - 1];
    // }
    E_last_ = E_;
  }
};

} // namespace openmc

#endif // OPENMC_PARTICLE_H
