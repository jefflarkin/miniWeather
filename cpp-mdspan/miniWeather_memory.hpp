#pragma once

#include "miniWeather_serial_memory.hpp"
#if defined(MINIWEATHER_KOKKOS)
#  include "miniWeather_kokkos_memory.hpp"
#endif
#if defined(MINIWEATHER_CUB)
#  include "miniWeather_cub_memory.hpp"
#endif

// All dynamic array allocation happens in the two functions
// make_unique_array_3d and make_unique_array_1d.
// Overload them for your execution space and memory space types.
// The functions take both execution and memory space in order to
// support stream-ordered allocation (e.g., cudaMallocAsync).

template<class ExecutionSpace, class MemorySpace>
using alloc_3d = decltype(make_unique_array_3d(ExecutionSpace{}, MemorySpace{}, 0, 0, 0));
template<class ExecutionSpace, class MemorySpace>
using alloc_1d = decltype(make_unique_array_1d(ExecutionSpace{}, MemorySpace{}, 0));
