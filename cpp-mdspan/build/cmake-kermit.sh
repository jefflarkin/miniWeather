#!/bin/bash

# This script only works with nvc++.
# It assumes that mpic++ finds nvc++, mpicc finds nvc (!= nvcc), etc.

PNETCDF_ROOT=/raid/mhoemmen/pkg/pnetcdf-1.14.0
PNETCDF_LDFLAGS="-L${PNETCDF_ROOT}/lib -lpnetcdf"
PNETCDF_CXXFLAGS="-I${PNETCDF_ROOT}/include"
SRC_ROOT=/raid/mhoemmen/src/miniWeather/cpp-mdspan

# "-stdpar": "Could not find librt library, needed by CUDA::cudart_static"
# Adding "-rt" to LDFLAGS didn't seem to help.

# Current version of stdpar code requires C++23,
# because it uses std::ranges::views::cartesian_product.
# nvc++ might not support that yet.
MINIWEATHER_ENABLE_STDPAR=ON

# We don't have ForEachInExtents yet with 25.1.
MINIWEATHER_ENABLE_CUB=OFF

KOKKOS_ROOT="/raid/mhoemmen/src/kokkos/kokkos"
#  -DFETCHCONTENT_SOURCE_DIR_Kokkos="${KOKKOS_ROOT}"
#  -DKokkos_ROOT="${KOKKOS_ROOT}"

LDFLAGS="${PNETCDF_LDFLAGS}" CXXFLAGS="${PNETCDF_CXXFLAGS}" cmake \
  -DCMAKE_CXX_COMPILER=mpic++ \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_Fortran_COMPILER=mpif90 \
  -DCMAKE_VERBOSE_MAKEFILE=ON \
  -DFETCHCONTENT_SOURCE_DIR_KOKKOS="${KOKKOS_ROOT}" \
  -DMINIWEATHER_ENABLE_STDPAR=${MINIWEATHER_ENABLE_STDPAR} \
  -DMINIWEATHER_ENABLE_CUB=${MINIWEATHER_ENABLE_CUB} \
  ${SRC_ROOT}

#  -DCMAKE_CXX_FLAGS="-stdpar" 
