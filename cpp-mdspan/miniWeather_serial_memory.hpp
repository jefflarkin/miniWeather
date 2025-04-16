#pragma once

#include <memory>

struct host_memory_space {};
struct host_serial_execution_policy {};

// Default behavior for host memory space is to use normal new and delete
// via std::make_unique.  You can override this by overloading the function
// for your execution space.
template<class ExecutionSpace>
std::unique_ptr<double[]>
make_unique_array_3d(ExecutionSpace, host_memory_space, int X, int Y, int Z) {
  return std::make_unique<double[]>(X * Y * Z);
}

template<class ExecutionSpace>
std::unique_ptr<double[]>
make_unique_array_1d(ExecutionSpace, host_memory_space, int X) {
  return std::make_unique<double[]>(X);
}

inline host_memory_space
default_memory_space(host_serial_execution_policy) {
  return {};
}
