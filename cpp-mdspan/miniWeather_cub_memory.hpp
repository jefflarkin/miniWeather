#pragma once

#if ! defined(MINIWEATHER_CUB)
#  error "CUB is not enabled"
#else
#  include "cuda.h"
#endif

struct cub_execution_policy {
  cudaStream_t stream = {};
};

cub_memory_space default_memory_space(cub_execution_policy) {
  return cub_memory_space{};
}

inline std::unique_ptr<double[], cub_deleter>
make_unique_array_3d(cub_execution_policy exec_space,
  cub_memory_space memory_space, int X, int Y, int Z)
{
  return kokkos_make_unique_array_3d(exec_space, memory_space, X, Y, Z);
}

inline std::unique_ptr<double[], cub_deleter>
make_unique_array_1d(cub_execution_policy exec_space,
  cub_memory_space memory_space, int X)
{
  return kokkos_make_unique_array_1d(exec_space, memory_space, X);
}
