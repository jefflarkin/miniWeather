//////////////////////////////////////////////////////////////////////////////////////////
// miniWeather
// Author: Matt Norman <normanmr@ornl.gov>  , Oak Ridge National Laboratory
// This code simulates dry, stratified, compressible, non-hydrostatic fluid flows
// For documentation, please see the attached documentation in the "documentation" folder
//
//////////////////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#include <mpi.h>
#include "pnetcdf.h"

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

std::unique_ptr<double[]>
make_unique_array_3d(host_memory_space, int X, int Y, int Z) {
  return std::make_unique<double[]>(X * Y * Z);
}
std::unique_ptr<double[]>
make_unique_array_1d(host_memory_space, int X) {
  return std::make_unique<double[]>(X);
}

template<class MemorySpace>
using alloc_3d = decltype(make_unique_array_3d(MemorySpace{}, 0, 0, 0));
template<class MemorySpace>
using alloc_1d = decltype(make_unique_array_1d(MemorySpace{}, 0));

using default_memory_space = host_memory_space;

auto make_unique_array_3d(int X, int Y, int Z) {
  return make_unique_array_3d(default_memory_space{}, X, Y, Z);
}
auto make_unique_array_1d(int X) {
  return make_unique_array_1d(default_memory_space{}, X);
}


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

  bool mainproc() const { return myrank == 0; } //Am I the main process (rank == 0)?
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

template<class MemorySpace>
init_result<MemorySpace> init(MemorySpace memory_space, int *argc , char ***argv);

void finalize();

struct test_case {
  double r;
  double u;
  double w;
  double t;
  double hr;
  double ht;
};

test_case injection(double x, double z);
test_case density_current(double x, double z);
test_case gravity_waves(double x, double z);
test_case thermal(double x, double z);
test_case collision(double x, double z);

test_case get_test_case(int data_spec, double x_, double z_) {
  if (data_spec == DATA_SPEC_COLLISION      ) { return collision(x_, z_); }
  if (data_spec == DATA_SPEC_THERMAL        ) { return thermal(x_, z_); }
  if (data_spec == DATA_SPEC_GRAVITY_WAVES  ) { return gravity_waves(x_, z_); }
  if (data_spec == DATA_SPEC_DENSITY_CURRENT) { return density_current(x_, z_); }
  if (data_spec == DATA_SPEC_INJECTION      ) { return injection(x_, z_); }
  assert(false);
  return test_case{};
}

struct r_t_pair {
  double r;
  double t;
};

r_t_pair hydro_const_theta(double z);
r_t_pair hydro_const_bvfreq(double z, double bv_freq0);
double sample_ellipse_cosine(double x, double z, double amp, double x0, double z0,
                             double xrad, double zrad);

template<class MemorySpace>
void output(view_3d_const state,
  const global_const_scalars& const_scalars,
  const global_const_arrays<MemorySpace>& const_arrays,
  global_scalars& scalars);
void ncwrap(int ierr, int line);
template<class MemorySpace>
void perform_timestep(view_3d state, view_3d state_tmp,
                      view_3d flux, view_3d tend,
                      const global_const_scalars& c_scalars,
                      const global_const_arrays<MemorySpace>& c_arrays,
                      global_scalars& scalars);
template<class MemorySpace>
void semi_discrete_step(view_3d_const state_init,
                        view_3d state_forcing,
                        view_3d state_out,
                        double dt /* not scalars.dt */,
                        direction dir, view_3d flux, view_3d tend,
                        const global_const_scalars& scalars,
                        const global_const_arrays<MemorySpace>& arrays);
template<class MemorySpace>
void compute_tendencies_x(view_3d_const state,
  view_3d flux, view_3d tend, double dt,
  const global_const_scalars& scalars,
  const global_const_arrays<MemorySpace>& arrays);
template<class MemorySpace>
void compute_tendencies_z(view_3d_const state,
  view_3d flux, view_3d tend, double dt,
  const global_const_scalars& scalars,
  const global_const_arrays<MemorySpace>& arrays);
template<class MemorySpace>
void set_halo_values_x(view_3d state,
  const global_const_scalars& scalars,
  const global_const_arrays<MemorySpace>& arrays);
template<class MemorySpace>
void set_halo_values_z(view_3d state,
  const global_const_scalars& scalars,
  const global_const_arrays<MemorySpace>& arrays);

struct reduction_result {
  double mass;
  double te;
};
template<class MemorySpace>
reduction_result reductions(view_3d_const state,
  const global_const_scalars& const_scalars,
  const global_const_arrays<MemorySpace>& const_arrays);

///////////////////////////////////////////////////////////////////////////////////////
// THE MAIN PROGRAM STARTS HERE
///////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  auto memory_space = default_memory_space{};
  auto [const_scalars, scalars, const_arrays, arrays] = init(memory_space, &argc , &argv );

  //Initial reductions for mass, kinetic energy, and total energy.
  //
  // mass0: initial domain total for mass
  // te0:   initial domain total for total energy
  auto [mass0, te0] = reductions(std::as_const(arrays).state(), const_scalars, const_arrays);
#if ! defined(NO_INFORM)
  if (const_scalars.mainproc()) {
    fprintf(stderr, "mass0: %le\n" , mass0);
    fprintf(stderr, "te0:   %le\n" , te0  );
  }
#endif

  //Output the initial state
  output(std::as_const(arrays).state(), const_scalars, const_arrays, scalars);

  ////////////////////////////////////////////////////
  // MAIN TIME STEP LOOP
  ////////////////////////////////////////////////////
  [[maybe_unused]] auto t1 = std::chrono::steady_clock::now();
  while (scalars.etime < sim_time) {
    // If the time step leads to exceeding the simulation time,
    // shorten it for the last step
    if (scalars.etime + scalars.dt > sim_time) {
      scalars.dt = sim_time - scalars.etime;
    }
    perform_timestep(arrays.state(), arrays.state_tmp(), arrays.flux(),
      arrays.tend(), const_scalars, const_arrays, scalars);
#if ! defined(NO_INFORM)
    if (const_scalars.mainproc()) {
      fprintf(stderr, "Elapsed Time: %lf / %lf\n", scalars.etime, sim_time);
    }
#endif
    //Update the elapsed time and output counter
    scalars.etime = scalars.etime + scalars.dt;
    scalars.output_counter = scalars.output_counter + scalars.dt;
    //If it's time for output, reset the counter, and do output
    if (scalars.output_counter >= output_freq) {
      scalars.output_counter = scalars.output_counter - output_freq;
      output(arrays.state(), const_scalars, const_arrays, scalars);
    }
  }
  [[maybe_unused]] auto t2 = std::chrono::steady_clock::now();
#if ! defined(NO_INFORM)
  if (const_scalars.mainproc()) {
    printf("CPU Time: %e s\n", std::chrono::duration<double>(t2-t1).count());
  }
#endif

  //Final reductions for mass, kinetic energy, and total energy
  auto [mass, te] = reductions(arrays.state(), const_scalars, const_arrays);
  if (const_scalars.mainproc()) {
    printf("d_mass: %le\n" , (mass - mass0)/mass0);
    printf("d_te:   %le\n" , (te   - te0  )/te0  );
  }

  finalize();
}

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
template<class MemorySpace>
void perform_timestep(view_3d state, view_3d state_tmp,
                      view_3d flux, view_3d tend,
                      const global_const_scalars& c_scalars,
                      const global_const_arrays<MemorySpace>& c_arrays,
                      global_scalars& scalars)
{
  const double dt = scalars.dt;
  if (scalars.direction_switch) {
    //x-direction first
    semi_discrete_step(state, state    , state_tmp, dt / 3, direction::X, flux, tend, c_scalars, c_arrays);
    semi_discrete_step(state, state_tmp, state_tmp, dt / 2, direction::X, flux, tend, c_scalars, c_arrays);
    semi_discrete_step(state, state_tmp, state    , dt / 1, direction::X, flux, tend, c_scalars, c_arrays);
    //z-direction second
    semi_discrete_step(state, state    , state_tmp, dt / 3, direction::Z, flux, tend, c_scalars, c_arrays);
    semi_discrete_step(state, state_tmp, state_tmp, dt / 2, direction::Z, flux, tend, c_scalars, c_arrays);
    semi_discrete_step(state, state_tmp, state    , dt / 1, direction::Z, flux, tend, c_scalars, c_arrays);
  } else {
    //z-direction second
    semi_discrete_step(state, state    , state_tmp, dt / 3, direction::Z, flux, tend, c_scalars, c_arrays);
    semi_discrete_step(state, state_tmp, state_tmp, dt / 2, direction::Z, flux, tend, c_scalars, c_arrays);
    semi_discrete_step(state, state_tmp, state    , dt / 1, direction::Z, flux, tend, c_scalars, c_arrays);
    //x-direction first
    semi_discrete_step(state, state    , state_tmp, dt / 3, direction::X, flux, tend, c_scalars, c_arrays);
    semi_discrete_step(state, state_tmp, state_tmp, dt / 2, direction::X, flux, tend, c_scalars, c_arrays);
    semi_discrete_step(state, state_tmp, state    , dt / 1, direction::X, flux, tend, c_scalars, c_arrays);
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
template<class MemorySpace>
void semi_discrete_step(view_3d_const state_init,
                        view_3d state_forcing,
                        view_3d state_out,
                        double dt /* not scalars.dt */,
                        direction dir, view_3d flux, view_3d tend,
                        const global_const_scalars& scalars,
                        const global_const_arrays<MemorySpace>& arrays)
{
  const int nx = scalars.nx;
  const int nz = scalars.nz;

  if (dir == direction::X) {
    //Set the halo values for this MPI task's fluid state in the x-direction
    set_halo_values_x(state_forcing, scalars, arrays);
    //Compute the time tendencies for the fluid state in the x-direction
    compute_tendencies_x(state_forcing, flux, tend, dt, scalars, arrays);
  } else if (dir == direction::Z) {
    //Set the halo values for this MPI task's fluid state in the z-direction
    set_halo_values_z(state_forcing, scalars, arrays);
    //Compute the time tendencies for the fluid state in the z-direction
    compute_tendencies_z(state_forcing, flux, tend, dt, scalars, arrays);
  }

  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Apply the tendencies to the fluid state

  {
    view_3d_const tend_c = tend;
    auto hy_dens_cell = arrays.hy_dens_cell();
    const int i_beg = scalars.i_beg;
    const int k_beg = scalars.k_beg;
    for (int ll = 0; ll < NUM_VARS; ++ll) {
      for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < nx; ++i) {
          if (data_spec_int == DATA_SPEC_GRAVITY_WAVES) {
            const double x = (i_beg + i+0.5)*dx;
            const double z = (k_beg + k+0.5)*dz;
            const double wpert = sample_ellipse_cosine(x, z, 0.01, xlen/8, 1000.0, 500.0, 500.0);
            tend(ID_WMOM, k, i) += wpert * hy_dens_cell[hs+k];
          }
          state_out(ll, k+hs, i+hs) = state_init(ll, k+hs, i+hs) + dt * tend_c(ll, k, i);
        }
      }
    }
  }
}


//Compute the time tendencies of the fluid state using forcing in the x-direction
//Since the halos are set in a separate routine, this will not require MPI
//First, compute the flux vector at each cell interface in the x-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
template<class MemorySpace>
void compute_tendencies_x(view_3d_const state,
  view_3d flux, view_3d tend, double dt,
  const global_const_scalars& scalars,
  const global_const_arrays<MemorySpace>& arrays)
{
  const int nx = scalars.nx;
  const int nz = scalars.nz;
  // Hyperviscosity coefficient
  const double hv_coef = -hv_beta * dx / (16*dt);
  auto hy_dens_cell = arrays.hy_dens_cell();
  auto hy_dens_theta_cell = arrays.hy_dens_theta_cell();

  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Compute fluxes in the x-direction for each cell
  for (int k = 0; k < nz; ++k) {
    for (int i = 0; i < nx+1; ++i) {
      //Use fourth-order interpolation from four cell averages
      //to compute the value at the interface in question
      std::array<double, NUM_VARS> d3_vals;
      std::array<double, NUM_VARS> vals;
      for (int ll = 0; ll < NUM_VARS; ++ll) {
        std::array<double, sten_size> stencil;
        for (int s = 0; s < sten_size; ++s) {
          stencil[s] = state(ll, k+hs, i+s);
        }
        //Fourth-order-accurate interpolation of the state
        vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
        //First-order-accurate interpolation of the third spatial derivative
        //of the state (for artificial viscosity)
        d3_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3];
      }

      //Compute density, u-wind, w-wind, potential temperature,
      //and pressure (r,u,w,t,p respectively)
      double r = vals[ID_DENS] + hy_dens_cell[k+hs];
      double u = vals[ID_UMOM] / r;
      double w = vals[ID_WMOM] / r;
      double t = ( vals[ID_RHOT] + hy_dens_theta_cell[k+hs] ) / r;
      double p = C0 * pow(r*t, gamm);

      //Compute the flux vector
      flux(ID_DENS, k, i) = r*u     - hv_coef*d3_vals[ID_DENS];
      flux(ID_UMOM, k, i) = r*u*u+p - hv_coef*d3_vals[ID_UMOM];
      flux(ID_WMOM, k, i) = r*u*w   - hv_coef*d3_vals[ID_WMOM];
      flux(ID_RHOT, k, i) = r*u*t   - hv_coef*d3_vals[ID_RHOT];
    }
  }

  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Use the fluxes to compute tendencies for each cell
  {
    view_3d_const flux_c = flux;
    for (int ll = 0; ll < NUM_VARS; ++ll) {
      for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < nx; ++i) {
          tend(ll, k, i) = -( flux_c(ll, k, i+1) - flux_c(ll, k, i) ) / dx;
        }
      }
    }
  }
}


//Compute the time tendencies of the fluid state using forcing in the z-direction
//Since the halos are set in a separate routine, this will not require MPI
//First, compute the flux vector at each cell interface in the z-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
template<class MemorySpace>
void compute_tendencies_z(view_3d_const state,
  view_3d flux, view_3d tend, double dt,
  const global_const_scalars& scalars,
  const global_const_arrays<MemorySpace>& arrays)
{
  const int nx = scalars.nx;
  const int nz = scalars.nz;
  // Hyperviscosity coefficient
  const double hv_coef = -hv_beta * dz / (16*dt);
  auto hy_dens_int = arrays.hy_dens_int();
  auto hy_dens_theta_int = arrays.hy_dens_theta_int();
  auto hy_pressure_int = arrays.hy_pressure_int();


  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Compute fluxes in the x-direction for each cell
  for (int k = 0; k < nz+1; ++k) {
    for (int i = 0; i < nx; ++i) {
      //Use fourth-order interpolation from four cell averages
      //to compute the value at the interface in question
      std::array<double, NUM_VARS> d3_vals;
      std::array<double, NUM_VARS> vals;
      for (int ll = 0; ll < NUM_VARS; ++ll) {
        std::array<double, sten_size> stencil;
        for (int s = 0; s < sten_size; ++s) {
          stencil[s] = state(ll, k+s, i+hs);
        }
        //Fourth-order-accurate interpolation of the state
        vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
        //First-order-accurate interpolation of the third spatial derivative
        //of the state
        d3_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3];
      }

      //Compute density, u-wind, w-wind, potential temperature,
      //and pressure (r,u,w,t,p respectively)
      double r = vals[ID_DENS] + hy_dens_int[k];
      double u = vals[ID_UMOM] / r;
      double w = vals[ID_WMOM] / r;
      double t = (vals[ID_RHOT] + hy_dens_theta_int[k]) / r;
      double p = C0 * pow(r * t, gamm) - hy_pressure_int[k];
      //Enforce vertical boundary condition and exact mass conservation
      if (k == 0 || k == nz) {
        w                = 0;
        d3_vals[ID_DENS] = 0;
      }

      //Compute the flux vector with hyperviscosity
      flux(ID_DENS, k, i) = r*w     - hv_coef*d3_vals[ID_DENS];
      flux(ID_UMOM, k, i) = r*w*u   - hv_coef*d3_vals[ID_UMOM];
      flux(ID_WMOM, k, i) = r*w*w+p - hv_coef*d3_vals[ID_WMOM];
      flux(ID_RHOT, k, i) = r*w*t   - hv_coef*d3_vals[ID_RHOT];
    }
  }

  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Use the fluxes to compute tendencies for each cell
  {
    view_3d_const flux_c = flux;
    for (int ll = 0; ll < NUM_VARS; ++ll) {
      for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < nx; ++i) {
          tend(ll, k, i) = -( flux_c(ll, k+1, i) - flux_c(ll, k, i) ) / dz;
          if (ll == ID_WMOM) {
            tend(ll, k, i) = tend(ll, k, i) - state(ID_DENS, k+hs, i+hs)*grav;
          }
        }
      }
    }
  }
}



//Set this MPI task's halo values in the x-direction. This routine will require MPI
template<class MemorySpace>
void set_halo_values_x(view_3d state,
  const global_const_scalars& scalars,
  const global_const_arrays<MemorySpace>& arrays)
{
  const int nx = scalars.nx;
  const int nz = scalars.nz;

  ////////////////////////////////////////////////////////////////////////
  // TODO: EXCHANGE HALO VALUES WITH NEIGHBORING MPI TASKS
  // (1) give    state(1:hs,1:nz,1:NUM_VARS)       to   my left  neighbor
  // (2) receive state(1-hs:0,1:nz,1:NUM_VARS)     from my left  neighbor
  // (3) give    state(nx-hs+1:nx,1:nz,1:NUM_VARS) to   my right neighbor
  // (4) receive state(nx+1:nx+hs,1:nz,1:NUM_VARS) from my right neighbor
  ////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////
  // DELETE THE SERIAL CODE BELOW AND REPLACE WITH MPI
  //////////////////////////////////////////////////////
  for (int ll = 0; ll < NUM_VARS; ++ll) {
    for (int k = 0; k < nz; ++k) {
      state(ll, k+hs, 0) = state(ll, k+hs, nx+hs-2);
      state(ll, k+hs, 1) = state(ll, k+hs, nx+hs-1);
      state(ll, k+hs, nx+hs) = state(ll, k+hs, hs);
      state(ll, k+hs, nx+hs+1) = state(ll, k+hs, hs+1);
    }
  }
  ////////////////////////////////////////////////////

  if (data_spec_int == DATA_SPEC_INJECTION) {
    if (scalars.myrank == 0) {
      auto hy_dens_cell = arrays.hy_dens_cell();
      auto hy_dens_theta_cell = arrays.hy_dens_theta_cell();
      const int k_beg = scalars.k_beg;
      for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < hs; ++i) {
          const double z = (k_beg + k+0.5)*dz;
          if (fabs(z-3*zlen/4) <= zlen/16) {
            state(ID_UMOM, k+hs, i) = (state(ID_DENS, k+hs, i) + hy_dens_cell[k+hs]) * 50.0;
            state(ID_RHOT, k+hs, i) = (state(ID_DENS, k+hs, i) + hy_dens_cell[k+hs]) * 298.0 -
              hy_dens_theta_cell[k+hs];
          }
        }
      }
    }
  }
}


//Set this MPI task's halo values in the z-direction. This does not require MPI because there is no MPI
//decomposition in the vertical direction
template<class MemorySpace>
void set_halo_values_z(view_3d state,
  const global_const_scalars& scalars,
  const global_const_arrays<MemorySpace>& arrays)
{
  const int nx = scalars.nx;
  const int nz = scalars.nz;
  auto hy_dens_cell = arrays.hy_dens_cell();

  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  for (int ll = 0; ll < NUM_VARS; ++ll) {
    for (int i = 0; i < nx+2*hs; ++i) {
      if (ll == ID_WMOM) {
        state(ll, 0, i) = 0.0;
        state(ll, 1, i) = 0.0;
        state(ll, nz+hs, i) = 0.0;
        state(ll, nz+hs+1, i) = 0.0;
      } else if (ll == ID_UMOM) {
        state(ll, 0, i) = state(ll, hs, i) / hy_dens_cell[hs] * hy_dens_cell[0];
        state(ll, 1, i) = state(ll, hs, i) / hy_dens_cell[hs] * hy_dens_cell[1];
        state(ll, nz+hs, i) = state(ll, nz+hs-1, i) / hy_dens_cell[nz+hs-1] * hy_dens_cell[nz+hs];
        state(ll, nz+hs+1, i) = state(ll, nz+hs-1, i) / hy_dens_cell[nz+hs-1] * hy_dens_cell[nz+hs+1];
      } else {
        state(ll, 0, i) = state(ll, hs, i);
        state(ll, 1, i) = state(ll, hs, i);
        state(ll, nz+hs, i) = state(ll, nz+hs-1, i);
        state(ll, nz+hs+1, i) = state(ll, nz+hs-1, i);
      }
    }
  }
}

template<class MemorySpace>
init_result<MemorySpace> init(MemorySpace memory_space, int *argc , char ***argv ) {
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

  //////////////////////////////////////////////////////////////////////////
  // Initialize the cell-averaged fluid state via Gauss-Legendre quadrature
  //////////////////////////////////////////////////////////////////////////
  for (int k = 0; k < nz+2*hs; ++k) {
    for (int i = 0; i < nx+2*hs; ++i) {
      //Initialize the state to zero
      for (int ll = 0; ll < NUM_VARS; ++ll) {
        state(ll, k, i) = 0.0;
      }
      //Use Gauss-Legendre quadrature to initialize a hydrostatic balance + temperature perturbation
      for (int kk = 0; kk < nqpoints; ++kk) {
        for (int ii = 0; ii < nqpoints; ++ii) {
          //Compute the x,z location within the global domain based on cell and quadrature index
          const double x = (i_beg + i-hs+0.5)*dx + (qpoints[ii]-0.5)*dx;
          const double z = (k_beg + k-hs+0.5)*dz + (qpoints[kk]-0.5)*dz;

          //Set the fluid state based on the user's specification
          auto [r, u, w, t, hr, ht] = get_test_case(data_spec_int, x, z);

          //Store into the fluid state array
          state(ID_DENS, k, i) = state(ID_DENS, k, i) + r                         * qweights[ii]*qweights[kk];
          state(ID_UMOM, k, i) = state(ID_UMOM, k, i) + (r+hr)*u                  * qweights[ii]*qweights[kk];
          state(ID_WMOM, k, i) = state(ID_WMOM, k, i) + (r+hr)*w                  * qweights[ii]*qweights[kk];
          state(ID_RHOT, k, i) = state(ID_RHOT, k, i) + ( (r+hr)*(t+ht) - hr*ht ) * qweights[ii]*qweights[kk];
        }
      }
      for (int ll = 0; ll < NUM_VARS; ++ll) {
        state_tmp(ll, k, i) = state(ll, k, i);
      }
    }
  }

  global_const_arrays gl_const_arrs(memory_space, nx, nz, hs);
  // Get nonconst views, so we can fill them in below.
  auto hy_dens_cell       = gl_const_arrs.hy_dens_cell();
  auto hy_dens_theta_cell = gl_const_arrs.hy_dens_theta_cell();
  auto hy_dens_int        = gl_const_arrs.hy_dens_int();
  auto hy_dens_theta_int  = gl_const_arrs.hy_dens_theta_int();
  auto hy_pressure_int    = gl_const_arrs.hy_pressure_int();

  //Compute the hydrostatic background state over vertical cell averages
  for (int k = 0; k < nz+2*hs; ++k) {
    hy_dens_cell[k] = 0.;
    hy_dens_theta_cell[k] = 0.;
    for (int kk = 0; kk < nqpoints; ++kk) {
      const double z = (k_beg + k-hs+0.5)*dz;
      //Set the fluid state based on the user's specification
      auto [r, u, w, t, hr, ht] = get_test_case(data_spec_int, 0.0, z);
      hy_dens_cell[k]       = hy_dens_cell[k]       + hr    * qweights[kk];
      hy_dens_theta_cell[k] = hy_dens_theta_cell[k] + hr*ht * qweights[kk];
    }
  }
  //Compute the hydrostatic background state at vertical cell interfaces
  for (int k = 0; k < nz+1; ++k) {
    const double z = (k_beg + k)*dz;
    auto [r, u, w, t, hr, ht] = get_test_case(data_spec_int, 0.0, z);
    hy_dens_int      [k] = hr;
    hy_dens_theta_int[k] = hr * ht;
    hy_pressure_int  [k] = C0 * pow(hr * ht, gamm);
  }

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


//This test case is initially balanced but injects fast, cold air from the left boundary near the model top
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
test_case injection(double x , double z) {
  auto [hr, ht] = hydro_const_theta(z);
  double r = 0.0;
  double t = 0.0;
  double u = 0.0;
  double w = 0.0;
  return {r, u, w, t, hr, ht};
}


//Initialize a density current (falling cold thermal that propagates along the model bottom)
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
test_case density_current(double x , double z) {
  auto [hr, ht] = hydro_const_theta(z);
  double r = 0.0;
  double t = sample_ellipse_cosine(x, z, -20.0, xlen/2, 5000.0, 4000.0, 2000.0);
  double u = 0.0;
  double w = 0.0;
  return {r, u, w, t, hr, ht};
}


//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
test_case gravity_waves(double x, double z) {
  auto [hr, ht] = hydro_const_bvfreq(z, 0.02);
  double r = 0.0;
  double t = 0.0;
  double u = 15.0;
  double w = 0.0;
  return {r, u, w, t, hr, ht};
}


//Rising thermal
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
test_case thermal(double x, double z) {
  auto [hr, ht] = hydro_const_theta(z);
  double r = 0.0;
  double t = sample_ellipse_cosine(x, z, 3.0, xlen/2,2000.0, 2000.0, 2000.0);
  double u = 0.0;
  double w = 0.0;
  return {r, u, w, t, hr, ht};
}


//Colliding thermals
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
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


//Establish hydrostatic balance using constant potential temperature (thermally neutral atmosphere)
//z is the input coordinate
//r and t are the output background hydrostatic density and potential temperature
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


//Output the fluid state (state) to a NetCDF file at a given elapsed model time (etime)
//The file I/O uses parallel-netcdf, the only external library required for this mini-app.
//If it's too cumbersome, you can comment the I/O out, but you'll miss out on some potentially cool graphics
template<class MemorySpace>
void output(view_3d_const state,
  const global_const_scalars& const_scalars,
  const global_const_arrays<MemorySpace>& const_arrays,
  global_scalars& scalars)
{
  const int nx = const_scalars.nx;
  const int nz = const_scalars.nz;

  int ncid, t_dimid, x_dimid, z_dimid, theta_varid, t_varid, dimids[3];
#if ! defined(MINIWEATHER_ONLY_OUTPUT_THETA)
  int dens_varid, uwnd_varid, wwnd_varid;
#endif
  MPI_Offset st1[1], ct1[1], st3[3], ct3[3];

  //Inform the user
  if (const_scalars.mainproc()) { fprintf(stderr, "*** OUTPUT ***\n"); }

  //Temporary arrays to hold density, u-wind, w-wind, and potential temperature (theta).
#if ! defined(MINIWEATHER_ONLY_OUTPUT_THETA)
  auto dens     = md::make_unique_mdarray<double>(nz, nx);
  auto uwnd     = md::make_unique_mdarray<double>(nz, nx);
  auto wwnd     = md::make_unique_mdarray<double>(nz, nx);
#endif
  auto theta    = md::make_unique_mdarray<double>(nz, nx);
  auto etimearr = std::make_unique<double[]>(1);

  // PNetCDF needs an MPI_Info object that is not MPI_INFO_NULL.
  // It's possible that earlier PNetCDF versions tolerated MPI_INFO_NULL.
  MPI_Info mpi_info;
  auto info_err = MPI_Info_create(&mpi_info);
  if (info_err != MPI_SUCCESS) {
    fprintf(stderr, "Error creating MPI Info object\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  //If the elapsed time is zero, create the file. Otherwise, open the file
  if (scalars.etime == 0) {
    //Create the file
    ncwrap( ncmpi_create( MPI_COMM_WORLD , "output.nc" , NC_CLOBBER , mpi_info , &ncid ) , __LINE__ );
    //Create the dimensions
    ncwrap( ncmpi_def_dim( ncid , "t" , (MPI_Offset) NC_UNLIMITED , &t_dimid ) , __LINE__ );
    ncwrap( ncmpi_def_dim( ncid , "x" , (MPI_Offset) nx_glob      , &x_dimid ) , __LINE__ );
    ncwrap( ncmpi_def_dim( ncid , "z" , (MPI_Offset) nz_glob      , &z_dimid ) , __LINE__ );
    //Create the variables
    dimids[0] = t_dimid;
    ncwrap( ncmpi_def_var( ncid , "t_var"     , NC_DOUBLE , 1 , dimids ,     &t_varid ) , __LINE__ );
    dimids[0] = t_dimid; dimids[1] = z_dimid; dimids[2] = x_dimid;
#if ! defined(MINIWEATHER_ONLY_OUTPUT_THETA)
    ncwrap( ncmpi_def_var( ncid , "dens"  , NC_DOUBLE , 3 , dimids ,  &dens_varid ) , __LINE__ );
    ncwrap( ncmpi_def_var( ncid , "uwnd"  , NC_DOUBLE , 3 , dimids ,  &uwnd_varid ) , __LINE__ );
    ncwrap( ncmpi_def_var( ncid , "wwnd"  , NC_DOUBLE , 3 , dimids ,  &wwnd_varid ) , __LINE__ );
#endif
    ncwrap( ncmpi_def_var( ncid , "theta" , NC_DOUBLE , 3 , dimids , &theta_varid ) , __LINE__ );
    //End "define" mode
    ncwrap( ncmpi_enddef( ncid ) , __LINE__ );
  } else {
    //Open the file
    ncwrap( ncmpi_open( MPI_COMM_WORLD , "output.nc" , NC_WRITE , mpi_info , &ncid ) , __LINE__ );
    //Get the variable IDs
#if ! defined(MINIWEATHER_ONLY_OUTPUT_THETA)
    ncwrap( ncmpi_inq_varid( ncid , "dens"  ,  &dens_varid ) , __LINE__ );
    ncwrap( ncmpi_inq_varid( ncid , "uwnd"  ,  &uwnd_varid ) , __LINE__ );
    ncwrap( ncmpi_inq_varid( ncid , "wwnd"  ,  &wwnd_varid ) , __LINE__ );
#endif
    ncwrap( ncmpi_inq_varid( ncid , "theta" , &theta_varid ) , __LINE__ );
    ncwrap( ncmpi_inq_varid( ncid , "t_var" ,     &t_varid ) , __LINE__ );
  }

  //Store perturbed values in the temp arrays for output

  auto hy_dens_cell       = const_arrays.hy_dens_cell();
  auto hy_dens_theta_cell = const_arrays.hy_dens_theta_cell();
  for (int k = 0; k < nz; ++k) {
    for (int i = 0; i < nx; ++i) {
#if ! defined(MINIWEATHER_ONLY_OUTPUT_THETA)
      dens(k, i) = state(ID_DENS, k+hs, i+hs);
      uwnd(k, i) = state(ID_UMOM, k+hs, i+hs) / (hy_dens_cell[k+hs] + state(ID_DENS, k+hs, i+hs));
      wwnd(k, i) = state(ID_WMOM, k+hs, i+hs) / (hy_dens_cell[k+hs] + state(ID_DENS, k+hs, i+hs));
#endif      
      theta(k, i) = (state(ID_RHOT, k+hs, i+hs) + hy_dens_theta_cell[k+hs]) /
                    (hy_dens_cell[k+hs] + state(ID_DENS, k+hs, i+hs)) -
                    hy_dens_theta_cell[k+hs] / hy_dens_cell[k+hs];
    }
  }

  //Write the grid data to file with all the processes writing collectively
  const int k_beg = const_scalars.k_beg;
  const int i_beg = const_scalars.i_beg;

  st3[0] = scalars.num_out; st3[1] = k_beg; st3[2] = i_beg;
  ct3[0] = 1;               ct3[1] = nz;    ct3[2] = nx;
#if ! defined(MINIWEATHER_ONLY_OUTPUT_THETA)      
  ncwrap( ncmpi_put_vara_double_all( ncid ,  dens_varid , st3 , ct3 , dens.get()  ) , __LINE__ );
  ncwrap( ncmpi_put_vara_double_all( ncid ,  uwnd_varid , st3 , ct3 , uwnd.get()  ) , __LINE__ );
  ncwrap( ncmpi_put_vara_double_all( ncid ,  wwnd_varid , st3 , ct3 , wwnd.get()  ) , __LINE__ );
#endif
  ncwrap( ncmpi_put_vara_double_all( ncid , theta_varid , st3 , ct3 , theta.get() ) , __LINE__ );

  //Only the main process needs to write the elapsed time
  //Begin "independent" write mode
  ncwrap( ncmpi_begin_indep_data(ncid) , __LINE__ );
  //write elapsed time to file
  if (const_scalars.mainproc()) {
    st1[0] = scalars.num_out;
    ct1[0] = 1;
    etimearr[0] = scalars.etime;
    ncwrap( ncmpi_put_vara_double( ncid , t_varid , st1 , ct1 , etimearr.get() ) , __LINE__ );
  }
  //End "independent" write mode
  ncwrap( ncmpi_end_indep_data(ncid) , __LINE__ );

  //Close the file
  ncwrap( ncmpi_close(ncid) , __LINE__ );

  (void) MPI_Info_free(&mpi_info);
  scalars.num_out++;
}


//Error reporting routine for the PNetCDF I/O
void ncwrap( int ierr , int line ) {
  if (ierr != NC_NOERR) {
    fprintf(stderr, "NetCDF Error at line: %d\n", line);
    fprintf(stderr, "%s\n", ncmpi_strerror(ierr));
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
}


void finalize() {
  (void) MPI_Finalize();
}


//Compute reduced quantities for error checking without resorting to the "ncdiff" tool
template<class MemorySpace>
reduction_result reductions(view_3d_const state,
  const global_const_scalars& const_scalars,
  const global_const_arrays<MemorySpace>& const_arrays)
{
  reduction_result result{0.0, 0.0};
  const int nx = const_scalars.nx;
  const int nz = const_scalars.nz;
  auto hy_dens_cell = const_arrays.hy_dens_cell();
  auto hy_dens_theta_cell = const_arrays.hy_dens_theta_cell();

  for (int k = 0; k < nz; ++k) {
    for (int i = 0; i < nx; ++i) {
      double r  =  state(ID_DENS, k+hs, i+hs) + hy_dens_cell[hs+k]; // Density
      double u  =  state(ID_UMOM, k+hs, i+hs) / r;                  // U-wind
      double w  =  state(ID_WMOM, k+hs, i+hs) / r;                  // W-wind
      double th = (state(ID_RHOT, k+hs, i+hs) + hy_dens_theta_cell[hs+k]) / r; // Potential Temperature (theta)
      double p  = C0 * pow(r * th, gamm);                           // Pressure
      double t  = th / pow(p0 / p, rd / cp);                        // Temperature
      double ke = r*(u*u+w*w);                                      // Kinetic Energy
      double ie = r*cv*t;                                           // Internal Energy
      result.mass += r        *dx*dz; // Accumulate domain mass
      result.te   += (ke + ie)*dx*dz; // Accumulate domain total energy
    }
  }
  std::array<double, 2> loc{result.mass, result.te};
  std::array<double, 2> glob{0.0, 0.0};
  int ierr = MPI_Allreduce(loc.data(), glob.data(), 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return reduction_result{
    .mass = glob[0],
    .te = glob[1]
  };
}


