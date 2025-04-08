#include "miniWeather_common.hpp"

std::unique_ptr<double[]>
make_unique_array_3d(host_serial_execution_policy, host_memory_space, int X, int Y, int Z) {
  return std::make_unique<double[]>(X * Y * Z);
}

std::unique_ptr<double[]>
make_unique_array_1d(host_serial_execution_policy, host_memory_space, int X) {
  return std::make_unique<double[]>(X);
}

void finalize() {
#if defined(MINIWEATHER_KOKKOS)
  Kokkos::finalize();
#endif
  (void) MPI_Finalize();
}
