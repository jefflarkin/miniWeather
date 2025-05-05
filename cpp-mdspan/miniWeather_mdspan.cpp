//////////////////////////////////////////////////////////////////////////////////////////
// miniWeather
// Author: Matt Norman <normanmr@ornl.gov>  , Oak Ridge National Laboratory
// This code simulates dry, stratified, compressible, non-hydrostatic fluid flows
// For documentation, please see the attached documentation in the "documentation" folder
//
//////////////////////////////////////////////////////////////////////////////////////////

#include "miniWeather_common.hpp"
#include "miniWeather_output.hpp"

#if defined(MINIWEATHER_CUB)
#  include "miniWeather_cub.hpp"
#elif defined(MINIWEATHER_KOKKOS)
#  include "miniWeather_kokkos.hpp"
#elif defined(MINIWEATHER_STDPAR)
#  include "miniWeather_stdpar.hpp"
#elif defined(MINIWEATHER_OPENACC)
// Doesn't exist yet
//#  include "miniWeather_openacc.hpp"
#else
#  include "miniWeather_serial.hpp"
#endif

// This needs to go after the above (execution policy - specific) headers.
#include "miniWeather_generic_algs.hpp"

// Implementations may, but are not required
// to specialize this for their execution space type(s).
template<class ExecutionSpace>
host_memory_space default_memory_space(ExecutionSpace) {
  return {};
}

#if defined(MINIWEATHER_CUB)
#  define MINIWEATHER_DEFAULT_EXECUTION_POLICY cub_execution_policy
#elif defined(MINIWEATHER_KOKKOS)
#  if defined(MINIWEATHER_KOKKOS_CUDA)
#    define MINIWEATHER_DEFAULT_EXECUTION_POLICY Kokkos::Cuda
#  elif defined(MINIWEATHER_KOKKOS_OPENACC)
#    define MINIWEATHER_DEFAULT_EXECUTION_POLICY Kokkos::Experimental::OpenACC
#  elif defined(MINIWEATHER_KOKKOS_SERIAL)
#    define MINIWEATHER_DEFAULT_EXECUTION_POLICY Kokkos::Serial
#  else
#    define MINIWEATHER_DEFAULT_EXECUTION_POLICY Kokkos::DefaultExecutionSpace
#  endif // MINIWEATHER_KOKKOS
#elif defined(MINIWEATHER_STDPAR)
#  define MINIWEATHER_DEFAULT_EXECUTION_POLICY stdpar_ranges_execution_policy
#elif defined(MINIWEATHER_OPENACC)
// FIXME define a separate one for OpenACC
#  define MINIWEATHER_DEFAULT_EXECUTION_POLICY host_serial_execution_policy
#elif defined(MINIWEATHER_SERIAL)
#  define MINIWEATHER_DEFAULT_EXECUTION_POLICY host_serial_execution_policy
#else
#  error "No default execution policy defined"
#endif

auto default_execution_policy() {
  return MINIWEATHER_DEFAULT_EXECUTION_POLICY{};
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
  auto exec_space = default_execution_policy();
  auto memory_space = default_memory_space(exec_space);
  auto [const_scalars, scalars, const_arrays, arrays] =
    init(exec_space, memory_space, &argc, &argv);

  //Initial reductions for mass, kinetic energy, and total energy.
  //
  // mass0: initial domain total for mass
  // te0:   initial domain total for total energy
  auto [mass0, te0] = reductions(exec_space,
    std::as_const(arrays).state(), const_scalars, const_arrays);
#if ! defined(NO_INFORM)
  if (const_scalars.mainproc()) {
    fprintf(stderr, "mass0: %le\n" , mass0);
    fprintf(stderr, "te0:   %le\n" , te0  );
  }
#endif

  //Output the initial state
  output(exec_space, std::as_const(arrays).state(),
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
    perform_timestep(exec_space,
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
      output(exec_space,
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
  auto [mass, te] = reductions(exec_space,
    arrays.state(), const_scalars, const_arrays);
  if (const_scalars.mainproc()) {
    printf("d_mass: %le\n" , (mass - mass0)/mass0);
    printf("d_te:   %le\n" , (te   - te0  )/te0  );
  }

  finalize();
}
