//////////////////////////////////////////////////////////////////////////////////////////
// miniWeather
// Author: Matt Norman <normanmr@ornl.gov>  , Oak Ridge National Laboratory
// This code simulates dry, stratified, compressible, non-hydrostatic fluid flows
// For documentation, please see the attached documentation in the "documentation" folder
//
//////////////////////////////////////////////////////////////////////////////////////////

#include "miniWeather_common.hpp"
#include "miniWeather_output.hpp"
#include "miniWeather_serial.hpp"

auto default_memory_space() {
  return host_memory_space{};
}

auto default_execution_policy() {
  return host_serial_execution_policy{};
}

// Intra-(MPI-process) parallelization needs to happen in the following functions.
//
// apply_tendencies_to_fluid_state
// set_halo_values_x
// set_halo_values_z
// compute_tendencies_x
// compute_tendencies_z
// initialize_cell_averaged_fluid_state
// compute_hydrostatic_background_state
// local_reductions

int main(int argc, char **argv) {
  auto exec_policy = default_execution_policy();
  auto memory_space = default_memory_space();
  auto [const_scalars, scalars, const_arrays, arrays] =
    init(exec_policy, memory_space, &argc , &argv);

  //Initial reductions for mass, kinetic energy, and total energy.
  //
  // mass0: initial domain total for mass
  // te0:   initial domain total for total energy
  auto [mass0, te0] = reductions(exec_policy,
    std::as_const(arrays).state(), const_scalars, const_arrays);
#if ! defined(NO_INFORM)
  if (const_scalars.mainproc()) {
    fprintf(stderr, "mass0: %le\n" , mass0);
    fprintf(stderr, "te0:   %le\n" , te0  );
  }
#endif

  //Output the initial state
  output(exec_policy, std::as_const(arrays).state(),
    const_scalars, const_arrays, scalars);

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
    perform_timestep(exec_policy,
      arrays.state(), arrays.state_tmp(), arrays.flux(),
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
      output(exec_policy,
        arrays.state(), const_scalars, const_arrays, scalars);
    }
  }
  [[maybe_unused]] auto t2 = std::chrono::steady_clock::now();
#if ! defined(NO_INFORM)
  if (const_scalars.mainproc()) {
    printf("CPU Time: %e s\n", std::chrono::duration<double>(t2-t1).count());
  }
#endif

  //Final reductions for mass, kinetic energy, and total energy
  auto [mass, te] = reductions(exec_policy,
    arrays.state(), const_scalars, const_arrays);
  if (const_scalars.mainproc()) {
    printf("d_mass: %le\n" , (mass - mass0)/mass0);
    printf("d_te:   %le\n" , (te   - te0  )/te0  );
  }

  finalize();
}
