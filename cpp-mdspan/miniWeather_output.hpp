#pragma once

#include "miniWeather_common.hpp"
#include "pnetcdf.h"

//Error reporting routine for the PNetCDF I/O
inline void ncwrap( int ierr , int line ) {
  if (ierr != NC_NOERR) {
    fprintf(stderr, "NetCDF Error at line: %d\n", line);
    fprintf(stderr, "%s\n", ncmpi_strerror(ierr));
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
}

//Output the fluid state (state) to a NetCDF file at a given elapsed model time (etime)
//The file I/O uses parallel-netcdf, the only external library required for this mini-app.
//If it's too cumbersome, you can comment the I/O out, but you'll miss out on some potentially cool graphics
template<class MemorySpace>
void output(
  host_serial_execution_policy exec_policy,
  view_3d_const state,
  const global_const_scalars& const_scalars,
  const global_const_arrays<MemorySpace>& const_arrays,
  global_scalars& scalars)
{
  const int nx = const_scalars.nx;
  const int nz = const_scalars.nz;

  int ncid, t_dimid, x_dimid, z_dimid, theta_varid, t_varid, dimids[3];
#if ! defined(MINIWEATHER_ONLY_OUTPUT_THETA)
  int dens_varid, uwnd_varid, wwnd_varid;
#endif
  MPI_Offset st1[1], ct1[1], st3[3], ct3[3];

  //Inform the user
  if (const_scalars.mainproc()) { fprintf(stderr, "*** OUTPUT ***\n"); }

  //Temporary arrays to hold density, u-wind, w-wind, and potential temperature (theta).
#if ! defined(MINIWEATHER_ONLY_OUTPUT_THETA)
  auto dens     = md::make_unique_mdarray<double>(nz, nx);
  auto uwnd     = md::make_unique_mdarray<double>(nz, nx);
  auto wwnd     = md::make_unique_mdarray<double>(nz, nx);
#endif
  auto theta    = md::make_unique_mdarray<double>(nz, nx);
  auto etimearr = std::make_unique<double[]>(1);

  // PNetCDF needs an MPI_Info object that is not MPI_INFO_NULL.
  // It's possible that earlier PNetCDF versions tolerated MPI_INFO_NULL.
  MPI_Info mpi_info;
  auto info_err = MPI_Info_create(&mpi_info);
  if (info_err != MPI_SUCCESS) {
    fprintf(stderr, "Error creating MPI Info object\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  //If the elapsed time is zero, create the file. Otherwise, open the file
  if (scalars.etime == 0) {
    //Create the file
    ncwrap( ncmpi_create( MPI_COMM_WORLD , "output.nc" , NC_CLOBBER , mpi_info , &ncid ) , __LINE__ );
    //Create the dimensions
    ncwrap( ncmpi_def_dim( ncid , "t" , (MPI_Offset) NC_UNLIMITED , &t_dimid ) , __LINE__ );
    ncwrap( ncmpi_def_dim( ncid , "x" , (MPI_Offset) nx_glob      , &x_dimid ) , __LINE__ );
    ncwrap( ncmpi_def_dim( ncid , "z" , (MPI_Offset) nz_glob      , &z_dimid ) , __LINE__ );
    //Create the variables
    dimids[0] = t_dimid;
    ncwrap( ncmpi_def_var( ncid , "t_var"     , NC_DOUBLE , 1 , dimids ,     &t_varid ) , __LINE__ );
    dimids[0] = t_dimid; dimids[1] = z_dimid; dimids[2] = x_dimid;
#if ! defined(MINIWEATHER_ONLY_OUTPUT_THETA)
    ncwrap( ncmpi_def_var( ncid , "dens"  , NC_DOUBLE , 3 , dimids ,  &dens_varid ) , __LINE__ );
    ncwrap( ncmpi_def_var( ncid , "uwnd"  , NC_DOUBLE , 3 , dimids ,  &uwnd_varid ) , __LINE__ );
    ncwrap( ncmpi_def_var( ncid , "wwnd"  , NC_DOUBLE , 3 , dimids ,  &wwnd_varid ) , __LINE__ );
#endif
    ncwrap( ncmpi_def_var( ncid , "theta" , NC_DOUBLE , 3 , dimids , &theta_varid ) , __LINE__ );
    //End "define" mode
    ncwrap( ncmpi_enddef( ncid ) , __LINE__ );
  } else {
    //Open the file
    ncwrap( ncmpi_open( MPI_COMM_WORLD , "output.nc" , NC_WRITE , mpi_info , &ncid ) , __LINE__ );
    //Get the variable IDs
#if ! defined(MINIWEATHER_ONLY_OUTPUT_THETA)
    ncwrap( ncmpi_inq_varid( ncid , "dens"  ,  &dens_varid ) , __LINE__ );
    ncwrap( ncmpi_inq_varid( ncid , "uwnd"  ,  &uwnd_varid ) , __LINE__ );
    ncwrap( ncmpi_inq_varid( ncid , "wwnd"  ,  &wwnd_varid ) , __LINE__ );
#endif
    ncwrap( ncmpi_inq_varid( ncid , "theta" , &theta_varid ) , __LINE__ );
    ncwrap( ncmpi_inq_varid( ncid , "t_var" ,     &t_varid ) , __LINE__ );
  }

  //Store perturbed values in the temp arrays for output

  auto hy_dens_cell       = const_arrays.hy_dens_cell();
  auto hy_dens_theta_cell = const_arrays.hy_dens_theta_cell();
  for (int k = 0; k < nz; ++k) {
    for (int i = 0; i < nx; ++i) {
#if ! defined(MINIWEATHER_ONLY_OUTPUT_THETA)
      dens(k, i) = state(ID_DENS, k+hs, i+hs);
      uwnd(k, i) = state(ID_UMOM, k+hs, i+hs) / (hy_dens_cell[k+hs] + state(ID_DENS, k+hs, i+hs));
      wwnd(k, i) = state(ID_WMOM, k+hs, i+hs) / (hy_dens_cell[k+hs] + state(ID_DENS, k+hs, i+hs));
#endif      
      theta(k, i) = (state(ID_RHOT, k+hs, i+hs) + hy_dens_theta_cell[k+hs]) /
                    (hy_dens_cell[k+hs] + state(ID_DENS, k+hs, i+hs)) -
                    hy_dens_theta_cell[k+hs] / hy_dens_cell[k+hs];
    }
  }

  //Write the grid data to file with all the processes writing collectively
  const int k_beg = const_scalars.k_beg;
  const int i_beg = const_scalars.i_beg;

  st3[0] = scalars.num_out; st3[1] = k_beg; st3[2] = i_beg;
  ct3[0] = 1;               ct3[1] = nz;    ct3[2] = nx;
#if ! defined(MINIWEATHER_ONLY_OUTPUT_THETA)      
  ncwrap( ncmpi_put_vara_double_all( ncid ,  dens_varid , st3 , ct3 , dens.get()  ) , __LINE__ );
  ncwrap( ncmpi_put_vara_double_all( ncid ,  uwnd_varid , st3 , ct3 , uwnd.get()  ) , __LINE__ );
  ncwrap( ncmpi_put_vara_double_all( ncid ,  wwnd_varid , st3 , ct3 , wwnd.get()  ) , __LINE__ );
#endif
  ncwrap( ncmpi_put_vara_double_all( ncid , theta_varid , st3 , ct3 , theta.get() ) , __LINE__ );

  //Only the main process needs to write the elapsed time
  //Begin "independent" write mode
  ncwrap( ncmpi_begin_indep_data(ncid) , __LINE__ );
  //write elapsed time to file
  if (const_scalars.mainproc()) {
    st1[0] = scalars.num_out;
    ct1[0] = 1;
    etimearr[0] = scalars.etime;
    ncwrap( ncmpi_put_vara_double( ncid , t_varid , st1 , ct1 , etimearr.get() ) , __LINE__ );
  }
  //End "independent" write mode
  ncwrap( ncmpi_end_indep_data(ncid) , __LINE__ );

  //Close the file
  ncwrap( ncmpi_close(ncid) , __LINE__ );

  (void) MPI_Info_free(&mpi_info);
  scalars.num_out++;
}
