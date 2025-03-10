#!/bin/bash

PNETCDF_ROOT=/raid/mhoemmen/pkg/pnetcdf-1.14.0
SRC_ROOT=/raid/mhoemmen/src/miniWeather/c
OPT_FLAGS="-g -O2"

cmake \
  -DCMAKE_CXX_COMPILER=mpic++ \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_Fortran_COMPILER=mpif90 \
  -DCXXFLAGS="${OPT_FLAGS} -I${PNETCDF_ROOT}/include" \
  -DLDFLAGS="-L${PNETCDF_ROOT}/lib -lpnetcdf" \
  -DNX=100 \
  -DNZ=50 \
  -DSIM_TIME=20 \
  -DOUT_FREQ=10 \
  ${SRC_ROOT}
