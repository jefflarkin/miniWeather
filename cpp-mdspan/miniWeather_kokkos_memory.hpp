#pragma once

#if defined(MINIWEATHER_KOKKOS)
#include "Kokkos_Core.hpp"
#else
#  error "Kokkos is not enabled"
#endif

#if defined(MINIWEATHER_KOKKOS_OPENACC)
#  if ! defined(KOKKOS_ENABLE_OPENACC)
#    error "Kokkos OpenACC is not enabled"
#  endif
#endif

#if defined(MINIWEATHER_KOKKOS_SERIAL)
#  if ! defined(KOKKOS_ENABLE_SERIAL)
#    error "Kokkos Serial is not enabled"
#  endif
#endif

#if defined(MINIWEATHER_KOKKOS_CUDA)
#  if ! defined(KOKKOS_ENABLE_CUDA)
#    error "Kokkos CUDA is not enabled"
#  endif
#endif

template<class MemorySpace>
  requires(Kokkos::is_memory_space_v<MemorySpace>)
struct kokkos_deleter {
  size_t alloc_size = 0;

  void operator() (void* ptr) const {
    MemorySpace{}.deallocate(ptr, alloc_size);
  }
};

template<class ExecutionSpace, class MemorySpace>
  requires(
    Kokkos::is_execution_space_v<ExecutionSpace> &&
    Kokkos::is_memory_space_v<MemorySpace>)
std::unique_ptr<double[], kokkos_deleter<MemorySpace>>
kokkos_make_unique_array_3d(ExecutionSpace exec_space,
  MemorySpace memory_space, int X, int Y, int Z)
{
  using deleter_type = kokkos_deleter<MemorySpace>;
  return std::unique_ptr<double[], deleter_type>(
    static_cast<double*>(memory_space.allocate(exec_space, X * Y * Z)),
    deleter_type{size_t(X) * size_t(Y) * size_t(Z)}
  );
}

template<class ExecutionSpace, class MemorySpace>
  requires(
    Kokkos::is_execution_space_v<ExecutionSpace> &&
    Kokkos::is_memory_space_v<MemorySpace>)
std::unique_ptr<double[], kokkos_deleter<MemorySpace>>
kokkos_make_unique_array_1d(ExecutionSpace exec_space,
  MemorySpace memory_space, int X)
{
  using deleter_type = kokkos_deleter<MemorySpace>;
  return std::unique_ptr<double[], deleter_type>(
    static_cast<double*>(memory_space.allocate(exec_space, X)),
    deleter_type{size_t(X)}
  );
}

#if defined(MINIWEATHER_KOKKOS_SERIAL)
inline std::unique_ptr<double[], kokkos_deleter<Kokkos::HostSpace>>
make_unique_array_3d(Kokkos::Serial exec_space,
  Kokkos::HostSpace memory_space, int X, int Y, int Z)
{
  return kokkos_make_unique_array_3d(exec_space, memory_space, X, Y, Z);
}

inline std::unique_ptr<double[], kokkos_deleter<Kokkos::HostSpace>>
make_unique_array_1d(Kokkos::Serial exec_space,
  Kokkos::HostSpace memory_space, int X)
{
  return kokkos_make_unique_array_1d(exec_space, memory_space, X);
}
#endif // MINIWEATHER_KOKKOS_SERIAL

#if defined(MINIWEATHER_KOKKOS_OPENACC)
inline std::unique_ptr<double[], kokkos_deleter<Kokkos::Experimental::OpenACCSpace>>
make_unique_array_3d(Kokkos::Experimental::OpenACC exec_space,
  Kokkos::Experimental::OpenACCSpace memory_space, int X, int Y, int Z)
{
  return kokkos_make_unique_array_3d(exec_space, memory_space, X, Y, Z);
}

inline std::unique_ptr<double[], kokkos_deleter<Kokkos::Experimental::OpenACCSpace>>
make_unique_array_1d(Kokkos::Experimental::OpenACC exec_space,
  Kokkos::Experimental::OpenACCSpace memory_space, int X)
{
  return kokkos_make_unique_array_1d(exec_space, memory_space, X);
}
#endif // MINIWEATHER_KOKKOS_OPENACC

#if defined(MINIWEATHER_KOKKOS_CUDA)
inline std::unique_ptr<double[], kokkos_deleter<Kokkos::CudaSpace>>
make_unique_array_3d(Kokkos::Cuda exec_space,
  Kokkos::CudaSpace memory_space, int X, int Y, int Z)
{
  return kokkos_make_unique_array_3d(exec_space, memory_space, X, Y, Z);
}

inline std::unique_ptr<double[], kokkos_deleter<Kokkos::CudaSpace>>
make_unique_array_1d(Kokkos::Cuda exec_space,
  Kokkos::CudaSpace memory_space, int X)
{
  return kokkos_make_unique_array_1d(exec_space, memory_space, X);
}
#endif // MINIWEATHER_KOKKOS_CUDA

#if defined(KOKKOS_ENABLE_SERIAL)
inline Kokkos::HostSpace
default_memory_space(Kokkos::Serial) {
  return {};
}
#endif

#if defined(KOKKOS_ENABLE_OPENACC)
inline Kokkos::Experimental::OpenACCSpace
default_memory_space(Kokkos::Experimental::OpenACC) {
  return {};
}
#endif

#if defined(KOKKOS_ENABLE_CUDA)
inline Kokkos::CudaSpace
default_memory_space(Kokkos::Cuda) {
  return {};
}
#endif
