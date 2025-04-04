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

#define MINIWEATHER_ONLY_OUTPUT_THETA 1

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
test_case injection(double x, double z);

//Initialize a density current (falling cold thermal that propagates along the model bottom)
test_case density_current(double x, double z);

test_case gravity_waves(double x, double z);

//Rising thermal
test_case thermal(double x, double z);

//Colliding thermals
test_case collision(double x, double z);

test_case get_test_case(int data_spec, double x_, double z_);

struct r_t_pair {
  double r;
  double t;
};

// Establish hydrostatic balance using constant potential temperature
// (thermally neutral atmosphere)
// z is the input coordinate
// r and t are the output background hydrostatic density and potential temperature
r_t_pair hydro_const_theta(double z);

//Establish hydrostatic balance using constant Brunt-Vaisala frequency
//z is the input coordinate
//bv_freq0 is the constant Brunt-Vaisala frequency
//r and t are the output background hydrostatic density and potential temperature
r_t_pair hydro_const_bvfreq(double z, double bv_freq0);

//Sample from an ellipse of a specified center, radius, and amplitude at a specified location
//x and z are input coordinates
//amp,x0,z0,xrad,zrad are input amplitude, center, and radius of the ellipse
double sample_ellipse_cosine(double x, double z, double amp, double x0, double z0,
                             double xrad, double zrad);

// Perform a single time step.
// Time steps are dimensionally split and
// use a simple low-storage three-stage Runge-Kutta time integrator.
// The dimensional splitting is a second-order-accurate alternating Strang splitting
// that alternates the order of directions each time step.
//
// The Runge-Kutta method used here is defined as follows:
//
// q*     = q[n] + dt/3 * rhs(q[n])
// q**    = q[n] + dt/2 * rhs(q*  )
// q[n+1] = q[n] + dt/1 * rhs(q** )
//
template<class ExecutionPolicy, class MemorySpace>
void perform_timestep(
  ExecutionPolicy exec_policy,
  view_3d state, view_3d state_tmp,
  view_3d flux, view_3d tend,
  const global_const_scalars& c_scalars,
  const global_const_arrays<MemorySpace>& c_arrays,
  global_scalars& scalars)
{
  const double dt = scalars.dt;
  if (scalars.direction_switch) {
    //x-direction first
    semi_discrete_step(exec_policy, state, state    , state_tmp, dt / 3, direction::X, flux, tend, c_scalars, c_arrays);
    semi_discrete_step(exec_policy, state, state_tmp, state_tmp, dt / 2, direction::X, flux, tend, c_scalars, c_arrays);
    semi_discrete_step(exec_policy, state, state_tmp, state    , dt / 1, direction::X, flux, tend, c_scalars, c_arrays);
    //z-direction second
    semi_discrete_step(exec_policy, state, state    , state_tmp, dt / 3, direction::Z, flux, tend, c_scalars, c_arrays);
    semi_discrete_step(exec_policy, state, state_tmp, state_tmp, dt / 2, direction::Z, flux, tend, c_scalars, c_arrays);
    semi_discrete_step(exec_policy, state, state_tmp, state    , dt / 1, direction::Z, flux, tend, c_scalars, c_arrays);
  } else {
    //z-direction second
    semi_discrete_step(exec_policy, state, state    , state_tmp, dt / 3, direction::Z, flux, tend, c_scalars, c_arrays);
    semi_discrete_step(exec_policy, state, state_tmp, state_tmp, dt / 2, direction::Z, flux, tend, c_scalars, c_arrays);
    semi_discrete_step(exec_policy, state, state_tmp, state    , dt / 1, direction::Z, flux, tend, c_scalars, c_arrays);
    //x-direction first
    semi_discrete_step(exec_policy, state, state    , state_tmp, dt / 3, direction::X, flux, tend, c_scalars, c_arrays);
    semi_discrete_step(exec_policy, state, state_tmp, state_tmp, dt / 2, direction::X, flux, tend, c_scalars, c_arrays);
    semi_discrete_step(exec_policy, state, state_tmp, state    , dt / 1, direction::X, flux, tend, c_scalars, c_arrays);
  }
  if (scalars.direction_switch) {
    scalars.direction_switch = 0;
  } else {
    scalars.direction_switch = 1;
  }
}

//Perform a single semi-discretized step in time with the form:
//state_out = state_init + dt * rhs(state_forcing)
//Meaning the step starts from state_init, computes the rhs using state_forcing,
//and stores the result in state_out
template<class ExecutionPolicy, class MemorySpace>
void semi_discrete_step(
  ExecutionPolicy exec_policy,
  view_3d_const state_init,
  view_3d state_forcing,
  view_3d state_out,
  double dt /* not scalars.dt */,
  direction dir, view_3d flux, view_3d tend,
  const global_const_scalars& scalars,
  const global_const_arrays<MemorySpace>& arrays)
{
  if (dir == direction::X) {
    //Set the halo values for this MPI task's fluid state in the x-direction
    set_halo_values_x(exec_policy, state_forcing, scalars, arrays);
    //Compute the time tendencies for the fluid state in the x-direction
    compute_tendencies_x(exec_policy, state_forcing, flux, tend, dt, scalars, arrays);
  } else if (dir == direction::Z) {
    //Set the halo values for this MPI task's fluid state in the z-direction
    set_halo_values_z(exec_policy, state_forcing, scalars, arrays);
    //Compute the time tendencies for the fluid state in the z-direction
    compute_tendencies_z(exec_policy, state_forcing, flux, tend, dt, scalars, arrays);
  }

  apply_tendencies_to_fluid_state(exec_policy, state_init, state_out, dt, tend, scalars, arrays);
}

template<class ExecutionPolicy, class MemorySpace>
init_result<MemorySpace> init(
  ExecutionPolicy exec_policy,
  MemorySpace memory_space,
  int *argc , char ***argv)
{
  (void) MPI_Init(argc,argv);

  /////////////////////////////////////////////////////////////
  // BEGIN MPI DUMMY SECTION
  // TODO: (1) GET NUMBER OF MPI RANKS
  //       (2) GET MY MPI RANK ID (RANKS ARE ZERO-BASED INDEX)
  //       (3) COMPUTE MY BEGINNING "I" INDEX (1-based index)
  //       (4) COMPUTE HOW MANY X-DIRECTION CELLS MY RANK HAS
  //       (5) FIND MY LEFT AND RIGHT NEIGHBORING RANK IDs
  /////////////////////////////////////////////////////////////
  int i_beg = 0;
  int nx = nx_glob;

  //////////////////////////////////////////////
  // END MPI DUMMY SECTION
  //////////////////////////////////////////////

  //Vertical direction isn't MPI-ized, so the rank's local values = the global values
  int k_beg = 0;
  int nz = nz_glob;
  int nranks = 1;
  int myrank = 0;
  int left_rank = 0;
  int right_rank = 0;
  bool mainproc = (myrank == 0);

  global_arrays gl_arrs(memory_space, nx, nz, hs);
  auto state = gl_arrs.state();
  auto state_tmp = gl_arrs.state_tmp();
  auto flux = gl_arrs.flux();
  auto tend = gl_arrs.tend();

  //Define the maximum stable time step based on an assumed maximum wind speed
  double dt = fmin(dx,dz) / max_speed * cfl;

  //If I'm the main process in MPI, display some grid information
  if (mainproc) {
    fprintf(stderr, "nx_glob, nz_glob: %d %d\n", nx_glob, nz_glob);
    fprintf(stderr, "dx,dz: %lf %lf\n",dx,dz);
    fprintf(stderr, "dt: %lf\n",dt);
  }
  //Want to make sure this info is displayed before further output
  (void) MPI_Barrier(MPI_COMM_WORLD);

  initialize_cell_averaged_fluid_state(exec_policy,
    state, state_tmp, nx, nz, i_beg, k_beg);

  global_const_arrays gl_const_arrs(memory_space, nx, nz, hs);
  // Get nonconst views, so we can fill them in below.
  auto hy_dens_cell       = gl_const_arrs.hy_dens_cell();
  auto hy_dens_theta_cell = gl_const_arrs.hy_dens_theta_cell();
  auto hy_dens_int        = gl_const_arrs.hy_dens_int();
  auto hy_dens_theta_int  = gl_const_arrs.hy_dens_theta_int();
  auto hy_pressure_int    = gl_const_arrs.hy_pressure_int();

  compute_hydrostatic_background_state(exec_policy,
    hy_dens_cell, hy_dens_theta_cell,
    hy_dens_int, hy_dens_theta_int, hy_pressure_int, nz, k_beg);

  return init_result{
    global_const_scalars{
#if defined(__cpp_designated_initializers)
      .nx = nx,
      .nz = nz,
      .i_beg = i_beg,
      .k_beg = k_beg,
      .nranks = nranks,
      .myrank = myrank,
      .left_rank = left_rank,
      .right_rank = right_rank
#else
      nx,
      nz,
      i_beg,
      k_beg,
      nranks,
      myrank,
      left_rank,
      right_rank
#endif
    },
    global_scalars{
#if defined(__cpp_designated_initializers)
      .dt = dt,
      .etime = 0.0,
      .output_counter = 0.0,
      .num_out = 0,
      .direction_switch = 1
#else
      dt,
      /* etime = */ 0.0,
      /* output_counter = */ 0.0,
      /* num_out = */ 0,
      /* direction_switch = */ 1
#endif
    },
    std::move(gl_const_arrs),
    std::move(gl_arrs)
  };
}

struct reduction_result {
  double mass;
  double te;
};

//Compute reduced quantities for error checking without resorting to the "ncdiff" tool
template<class ExecutionPolicy, class MemorySpace>
reduction_result reductions(
  ExecutionPolicy exec_policy,
  view_3d_const state,
  const global_const_scalars& const_scalars,
  const global_const_arrays<MemorySpace>& const_arrays)
{
  reduction_result result = local_reductions(exec_policy,
    state, const_scalars, const_arrays);
  std::array<double, 2> loc{result.mass, result.te};
  std::array<double, 2> glob{0.0, 0.0};
  int ierr = MPI_Allreduce(loc.data(), glob.data(), 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return reduction_result{
    .mass = glob[0],
    .te = glob[1]
  };
}

void finalize();
