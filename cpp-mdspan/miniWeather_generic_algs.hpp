#pragma once

#include "miniWeather_common.hpp"

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



