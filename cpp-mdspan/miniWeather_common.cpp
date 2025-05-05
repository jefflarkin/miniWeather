#include "miniWeather_common.hpp"

void finalize() {
#if defined(MINIWEATHER_KOKKOS)
  Kokkos::finalize();
#endif
  (void) MPI_Finalize();
}
