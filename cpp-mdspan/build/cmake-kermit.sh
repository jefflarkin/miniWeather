#!/bin/bash

PNETCDF_ROOT=/raid/mhoemmen/pkg/pnetcdf-1.14.0
SRC_ROOT=/raid/mhoemmen/src/miniWeather/cpp-mdspan
mdspan_ROOT=/raid/mhoemmen/src/kokkos/mdspan
# Setting -Dmdspan_ROOT="${mdspan_ROOT}" has no effect on FetchContent.

LDFLAGS="-L${PNETCDF_ROOT}/lib -lpnetcdf" CXXFLAGS="-I${PNETCDF_ROOT}/include" cmake \
  -DCMAKE_CXX_COMPILER=mpic++ \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_Fortran_COMPILER=mpif90 \
  -DCMAKE_VERBOSE_MAKEFILE=ON \
  ${SRC_ROOT}
