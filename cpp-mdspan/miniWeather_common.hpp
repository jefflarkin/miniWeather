//////////////////////////////////////////////////////////////////////////////////////////
// miniWeather
// Author: Matt Norman <normanmr@ornl.gov>  , Oak Ridge National Laboratory
// This code simulates dry, stratified, compressible, non-hydrostatic fluid flows
// For documentation, please see the attached documentation in the "documentation" folder
//
//////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <mpi.h>

// nvc++ in 25.1 doesn't like including this header.
//#include "cub/cub.cuh"

#include "mdspan/mdspan.hpp"
#include "unique_mdarray.hpp"

#if defined(MINIWEATHER_KOKKOS)
#  include "Kokkos_Core.hpp"
#  define MINIWEATHER_INLINE_FUNCTION KOKKOS_INLINE_FUNCTION
#else
#  define MINIWEATHER_INLINE_FUNCTION inline
#endif

constexpr double pi        = 3.14159265358979323846264338327;   //Pi
constexpr double grav      = 9.8;                               //Gravitational acceleration (m / s^2)
constexpr double cp        = 1004.;                             //Specific heat of dry air at constant pressure
constexpr double cv        = 717.;                              //Specific heat of dry air at constant volume
constexpr double rd        = 287.;                              //Dry air constant for equation of state (P=rho*rd*T)
constexpr double p0        = 1.e5;                              //Standard pressure at the surface in Pascals
constexpr double C0        = 27.5629410929725921310572974482;   //Constant to translate potential temperature into pressure (P=C0*(rho*theta)**gamma)
constexpr double gamm      = 1.40027894002789400278940027894;   //gamma=cp/Rd , have to call this gamm because "gamma" is taken (I hate C so much)
//Define domain and stability-related constants
constexpr double xlen      = 2.e4;    //Length of the domain in the x-direction (meters)
constexpr double zlen      = 1.e4;    //Length of the domain in the z-direction (meters)
constexpr double hv_beta   = 0.05;    //How strong to diffuse the solution: hv_beta \in [0:1]
constexpr double cfl       = 1.50;    //"Courant, Friedrichs, Lewy" number (for numerical stability)
constexpr double max_speed = 450;     //Assumed maximum wave speed during the simulation (speed of sound + speed of wind) (meter / sec)
constexpr int hs        = 2;          //"Halo" size: number of cells beyond the MPI tasks's domain needed for a full "stencil" of information for reconstruction
constexpr int sten_size = 4;          //Size of the stencil used for interpolation

//Parameters for indexing and flags
constexpr int NUM_VARS = 4;           //Number of fluid state variables
constexpr int ID_DENS  = 0;           //index for density ("rho")
constexpr int ID_UMOM  = 1;           //index for momentum in the x-direction ("rho * u")
constexpr int ID_WMOM  = 2;           //index for momentum in the z-direction ("rho * w")
constexpr int ID_RHOT  = 3;           //index for density * potential temperature ("rho * theta")

enum class direction { X, Z };

constexpr int DATA_SPEC_COLLISION       = 1;
constexpr int DATA_SPEC_THERMAL         = 2;
constexpr int DATA_SPEC_GRAVITY_WAVES   = 3;
constexpr int DATA_SPEC_DENSITY_CURRENT = 5;
constexpr int DATA_SPEC_INJECTION       = 6;

constexpr int nqpoints = 3;
constexpr double qpoints [] = { 0.112701665379258311482073460022E0 , 0.500000000000000000000000000000E0 , 0.887298334620741688517926539980E0 };
constexpr double qweights[] = { 0.277777777777777777777777777779E0 , 0.444444444444444444444444444444E0 , 0.277777777777777777777777777779E0 };

///////////////////////////////////////////////////////////////////////////////////////
// BEGIN USER-CONFIGURABLE PARAMETERS
///////////////////////////////////////////////////////////////////////////////////////
//The x-direction length is twice as long as the z-direction length
//So, you'll want to have nx_glob be twice as large as nz_glob

int    constexpr nz_glob       = 50;            //Number of total cells in the z-direction
int    constexpr nx_glob       = 2 * nz_glob;    //Number of total cells in the x-direction
double constexpr sim_time      = 1000.0;      //How many seconds to run the simulation
double constexpr output_freq   = 10.0;      //How frequently to output data to file (in seconds)
int    constexpr data_spec_int = DATA_SPEC_THERMAL;     //How to initialize the data
double constexpr dx            = xlen / nx_glob; // grid spacing in the x-direction
double constexpr dz            = zlen / nz_glob; // grid spacing in the x-direction
///////////////////////////////////////////////////////////////////////////////////////
// END USER-CONFIGURABLE PARAMETERS
///////////////////////////////////////////////////////////////////////////////////////

// NUM_VARS is a compile-time constant, so we bake it into the extents type.
using extents_3d =    md::extents<int, NUM_VARS, md::dynamic_extent, md::dynamic_extent>;
using view_3d =       md::mdspan<double,       extents_3d, md::layout_right>;
using view_3d_const = md::mdspan<const double, extents_3d, md::layout_right>;
using extents_1d =    md::extents<int, md::dynamic_extent>; // a.k.a. dims<1, int>;
using view_1d =       md::mdspan<double,       extents_1d, md::layout_right>;
using view_1d_const = md::mdspan<const double, extents_1d, md::layout_right>;

// All dynamic array allocation happens here.
// Deallocation other than through `delete [] ptr` would happen
// through a custom Deleter (second template argument of `unique_ptr`).
//
// "auto" return type makes it easier for allocation
// to depend on the build configuration.

struct host_memory_space {};
struct host_serial_execution_policy {};

std::unique_ptr<double[]>
make_unique_array_3d(host_memory_space, int X, int Y, int Z);

std::unique_ptr<double[]>
make_unique_array_1d(host_memory_space, int X);

template<class MemorySpace>
using alloc_3d = decltype(make_unique_array_3d(MemorySpace{}, 0, 0, 0));
template<class MemorySpace>
using alloc_1d = decltype(make_unique_array_1d(MemorySpace{}, 0));

// Variables that are set once in init and remain read-only throughout the simulation.
struct global_const_scalars {
  int nx = nx_glob;
  int nz = nz_glob; //Number of local grid cells in the x- and z- dimensions for this MPI task
  int i_beg = 0;
  int k_beg = 0;       //beginning index in the x- and z-directions for this MPI task

  int nranks = 1;
  int myrank = 0;        //Number of MPI ranks and my rank id
  int left_rank = 0;
  int right_rank = 0; //MPI Rank IDs that exist to my left and right in the global domain

  inline bool mainproc() const { return myrank == 0; } //Am I the main process (rank == 0)?
};

struct global_scalars {
  // Model time step (seconds).  The last time step might shorten this.
  double dt;
  double etime = 0.0;          //Elapsed model time
  double output_counter = 0.0; //Helps determine when it's time to do output
  int num_out = 0;             //Number of outputs performed
  int direction_switch = 1;    //Switch to alternate the order of directions
};

// Arrays that are allocated and filled in init and never changed after that.
template<class MemorySpace>
class global_const_arrays {
public:
  global_const_arrays(MemorySpace memory_space, int nx, int nz, int hs) :
    nx_(nx),
    nz_(nz),
    hs_(hs),
    hy_dens_cell_      (make_unique_array_1d(memory_space, nz+2*hs)),
    hy_dens_theta_cell_(make_unique_array_1d(memory_space, nz+2*hs)),
    hy_dens_int_       (make_unique_array_1d(memory_space, nz+1)),
    hy_dens_theta_int_ (make_unique_array_1d(memory_space, nz+1)),
    hy_pressure_int_   (make_unique_array_1d(memory_space, nz+1))
  {}

  // Const views exist for all use after init.
  view_1d_const hy_dens_cell() const {
    return view_1d_const{hy_dens_cell_.get(), nz_ + 2 * hs_};
  }
  view_1d_const hy_dens_theta_cell() const {
    return view_1d_const{hy_dens_theta_cell_.get(), nz_ + 2 * hs_};
  }
  view_1d_const hy_dens_int() const {
    return view_1d_const{hy_dens_int_.get(), nz_ + 1};
  }
  view_1d_const hy_dens_theta_int() const {
    return view_1d_const{hy_dens_theta_int_.get(), nz_ + 1};
  }
  view_1d_const hy_pressure_int() const {
    return view_1d_const{hy_pressure_int_.get(), nz_ + 1};
  }

  // Nonconst views exist for init.
  view_1d hy_dens_cell() {
    return view_1d{hy_dens_cell_.get(), nz_ + 2 * hs_};
  }
  view_1d hy_dens_theta_cell() {
    return view_1d{hy_dens_theta_cell_.get(), nz_ + 2 * hs_};
  }
  view_1d hy_dens_int() {
    return view_1d{hy_dens_int_.get(), nz_ + 1};
  }
  view_1d hy_dens_theta_int() {
    return view_1d{hy_dens_theta_int_.get(), nz_ + 1};
  }
  view_1d hy_pressure_int() {
    return view_1d{hy_pressure_int_.get(), nz_ + 1};
  }

private:
  int nx_, nz_, hs_;
  alloc_1d<MemorySpace> hy_dens_cell_;       //hydrostatic density (vert cell avgs).   Dimensions: (1-hs:nz+hs)
  alloc_1d<MemorySpace> hy_dens_theta_cell_; //hydrostatic rho*t (vert cell avgs).     Dimensions: (1-hs:nz+hs)
  alloc_1d<MemorySpace> hy_dens_int_;        //hydrostatic density (vert cell interf). Dimensions: (1:nz+1)
  alloc_1d<MemorySpace> hy_dens_theta_int_;  //hydrostatic rho*t (vert cell interf).   Dimensions: (1:nz+1)
  alloc_1d<MemorySpace> hy_pressure_int_;    //hydrostatic press (vert cell interf).   Dimensions: (1:nz+1)
};

// Arrays that are allocated in init and updated throughout the simulation.
//
// C indexing seems to prefer the extents in reverse order.
// Respecting that also avoids divergence from the Python version.
// This means that the mdspan must be layout_right; the intent appears
// to be for C code to use row-major storage, but with Fortran ordering.
template<class MemorySpace>
class global_arrays {
public:
  global_arrays(MemorySpace memory_space, int nx, int nz, int hs) :
    nx_(nx),
    nz_(nz),
    hs_(hs),
    state_    (make_unique_array_3d(memory_space, NUM_VARS, nz+2*hs, nx+2*hs)),
    state_tmp_(make_unique_array_3d(memory_space, NUM_VARS, nz+2*hs, nx+2*hs)),
    flux_     (make_unique_array_3d(memory_space, NUM_VARS, nz+1, nx+1)),
    tend_     (make_unique_array_3d(memory_space, NUM_VARS, nz, nx))
  {}

  // The current model for member functions that get a view of an array
  // is for the const-ness of the global_arrays object to determine
  // whether the view is a view-of-const or view-of-nonconst.
  // We might consider a different model where users explicitly declare
  // access intent (read-only, write-only, or read-write) at the point of use.
  view_3d state() {
    // The various allocations have related dimensions that depend on
    // just a few metadata (NUM_VARS, nz, nx, and hs).
    // Storing extents for each allocation would duplicate metadata storage.
    // Instead, we use flat allocations and construct layout mappings on the fly
    // in the member functions that return (mdspan) views.
    return view_3d{state_.get(), NUM_VARS, nz_ + 2 * hs_, nx_ + 2 * hs_};
  }
  view_3d state_tmp() {
    return view_3d{state_tmp_.get(), NUM_VARS, nz_ + 2 * hs_, nx_ + 2 * hs_};
  }
  view_3d flux() {
    return view_3d{flux_.get(), NUM_VARS, nz_ + 1, nx_ + 1};
  }
  view_3d tend() {
    return view_3d{tend_.get(), NUM_VARS, nz_, nx_};
  }

  view_3d_const state() const {
    // The various allocations have related dimensions that depend on
    // just a few metadata (NUM_VARS, nz, nx, and hs).
    // Storing extents for each allocation would duplicate metadata storage.
    // Instead, we use flat allocations and construct layout mappings on the fly
    // in the member functions that return (mdspan) views.
    return view_3d_const{state_.get(), NUM_VARS, nz_ + 2 * hs_, nx_ + 2 * hs_};
  }

private:
  int nx_, nz_, hs_;
  alloc_3d<MemorySpace> state_;     // Fluid state.             Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
  alloc_3d<MemorySpace> state_tmp_; // Fluid state.             Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
  alloc_3d<MemorySpace> flux_;      // Cell interface fluxes.   Dimensions: (nx+1,nz+1,NUM_VARS)
  alloc_3d<MemorySpace> tend_;      // Fluid state tendencies.  Dimensions: (nx,nz,NUM_VARS)
};

template<class MemorySpace>
struct init_result {
  global_const_scalars const_scalars;
  global_scalars scalars;
  global_const_arrays<MemorySpace> const_arrays;
  global_arrays<MemorySpace> arrays;
};

struct r_t_pair {
  double r;
  double t;
};

// Establish hydrostatic balance using constant potential temperature
// (thermally neutral atmosphere)
// z is the input coordinate
// r and t are the output background hydrostatic density and potential temperature
MINIWEATHER_INLINE_FUNCTION
r_t_pair hydro_const_theta(double z) {
  const double theta0 = 300.;  //Background potential temperature
  const double exner0 = 1.;    //Surface-level Exner pressure
  double       p,exner,rt;
  //Establish hydrostatic balance first using Exner pressure
  double t = theta0;                         //Potential Temperature at z
  exner = exner0 - grav * z / (cp * theta0); //Exner pressure at z
  p = p0 * pow(exner,(cp/rd));               //Pressure at z
  rt = pow((p / C0),(1. / gamm));            //rho*theta at z
  double r = rt / t;                         //Density at z

  return {r, t};
}

//Establish hydrostatic balance using constant Brunt-Vaisala frequency
//z is the input coordinate
//bv_freq0 is the constant Brunt-Vaisala frequency
//r and t are the output background hydrostatic density and potential temperature
MINIWEATHER_INLINE_FUNCTION
r_t_pair hydro_const_bvfreq(double z, double bv_freq0) {
  const double theta0 = 300.;  //Background potential temperature
  const double exner0 = 1.;    //Surface-level Exner pressure
  double       p, exner, rt;
  double t = theta0 * exp( bv_freq0*bv_freq0 / grav * z );                                    //Pot temp at z
  exner = exner0 - grav*grav / (cp * bv_freq0*bv_freq0) * (t - theta0) / (t * theta0); //Exner pressure at z
  p = p0 * pow(exner,(cp/rd));                                                         //Pressure at z
  rt = pow((p / C0), (1. / gamm));                                                  //rho*theta at z
  double r = rt / t;                                                                          //Density at z

  return {r, t};
}

//Sample from an ellipse of a specified center, radius, and amplitude at a specified location
//x and z are input coordinates
//amp,x0,z0,xrad,zrad are input amplitude, center, and radius of the ellipse
MINIWEATHER_INLINE_FUNCTION
double sample_ellipse_cosine( double x , double z , double amp , double x0 , double z0 , double xrad , double zrad ) {
  double dist;
  //Compute distance from bubble center
  dist = sqrt( ((x-x0)/xrad)*((x-x0)/xrad) + ((z-z0)/zrad)*((z-z0)/zrad) ) * pi / 2.0;
  //If the distance from bubble center is less than the radius, create a cos**2 profile
  if (dist <= pi / 2.0) {
    return amp * pow(cos(dist), 2.0);
  } else {
    return 0.;
  }
}

struct test_case {
  double r;
  double u;
  double w;
  double t;
  double hr;
  double ht;
};

// For the various test case functions, x and z are input coordinates at which to sample;
// r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location; and
// hr and ht are output background hydrostatic density and potential temperature at that location.

//This test case is initially balanced but injects fast, cold air from the left boundary near the model top
MINIWEATHER_INLINE_FUNCTION
test_case injection(double x , double z) {
  auto [hr, ht] = hydro_const_theta(z);
  double r = 0.0;
  double t = 0.0;
  double u = 0.0;
  double w = 0.0;
  return {r, u, w, t, hr, ht};
}

//Initialize a density current (falling cold thermal that propagates along the model bottom)
MINIWEATHER_INLINE_FUNCTION
test_case density_current(double x , double z) {
  auto [hr, ht] = hydro_const_theta(z);
  double r = 0.0;
  double t = sample_ellipse_cosine(x, z, -20.0, xlen/2, 5000.0, 4000.0, 2000.0);
  double u = 0.0;
  double w = 0.0;
  return {r, u, w, t, hr, ht};
}

MINIWEATHER_INLINE_FUNCTION
test_case gravity_waves(double x, double z) {
  auto [hr, ht] = hydro_const_bvfreq(z, 0.02);
  double r = 0.0;
  double t = 0.0;
  double u = 15.0;
  double w = 0.0;
  return {r, u, w, t, hr, ht};
}

//Rising thermal
MINIWEATHER_INLINE_FUNCTION
test_case thermal(double x, double z) {
  auto [hr, ht] = hydro_const_theta(z);
  double r = 0.0;
  double t = sample_ellipse_cosine(x, z, 3.0, xlen/2,2000.0, 2000.0, 2000.0);
  double u = 0.0;
  double w = 0.0;
  return {r, u, w, t, hr, ht};
}

//Colliding thermals
MINIWEATHER_INLINE_FUNCTION
test_case collision(double x , double z) {
  auto [hr, ht] = hydro_const_theta(z);
  double r = 0.0;
  double t = 0.0;
  double u = 0.0;
  double w = 0.0;
  t = t + sample_ellipse_cosine(x, z,  20.0, xlen/2,2000.0, 2000.0, 2000.0);
  t = t + sample_ellipse_cosine(x, z, -20.0, xlen/2,8000.0, 2000.0, 2000.0);
  return {r, u, w, t, hr, ht};
}

MINIWEATHER_INLINE_FUNCTION
test_case get_test_case(int data_spec, double x, double z) {
  if (data_spec == DATA_SPEC_COLLISION      ) { return collision(x, z); }
  if (data_spec == DATA_SPEC_THERMAL        ) { return thermal(x, z); }
  if (data_spec == DATA_SPEC_GRAVITY_WAVES  ) { return gravity_waves(x, z); }
  if (data_spec == DATA_SPEC_DENSITY_CURRENT) { return density_current(x, z); }
  if (data_spec == DATA_SPEC_INJECTION      ) { return injection(x, z); }
  assert(false);
  return test_case{};
}

struct reduction_result {
  double mass;
  double te;
};

void finalize();
