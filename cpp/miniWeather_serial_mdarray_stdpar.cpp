
//////////////////////////////////////////////////////////////////////////////////////////
// miniWeather
// Author: Matt Norman <normanmr@ornl.gov>  , Oak Ridge National Laboratory
//         Jeff Larkin <jlarkin@nvidia.com> , NVIDIA Corporation
// This code simulates dry, stratified, compressible, non-hydrostatic fluid flows
// For documentation, please see the attached documentation in the "documentation" folder
//
//////////////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <ctime>
#include <utility>
#include <tuple>
#include <algorithm>
#include <numeric>
#include <execution>
#include <ranges>
#include "const.h"
#include "pnetcdf.h"
#include <chrono>

// These should become unnecessary with the addition of cartesian_product
constexpr std::tuple<int,int> idx2d(int idx, int nx) { return {idx%nx, idx/nx}; }
constexpr std::tuple<int,int,int> idx3d(int idx, int nx, int nz) { return {idx%nx, (idx/nx)%nz, idx / (nx*nz)}; }

#include <experimental/mdarray>
namespace stdex = std::experimental;

template<class ValueType, size_t Rank>
using array_container = stdex::mdarray<ValueType, stdex::dextents<size_t, Rank>>;
using real1d_container = array_container<real, 1>;
using real2d_container = array_container<real, 2>;
using real3d_container = array_container<real, 3>;
using double1d_container = array_container<double, 1>;
using double2d_container = array_container<double, 2>;
using double3d_container = array_container<double, 3>;

template<class ElementType, size_t Rank>
using array_view = stdex::mdspan<ElementType, stdex::dextents<size_t, Rank>>;
using real1d_view = array_view<real, 1>;
using real2d_view = array_view<real, 2>;
using real3d_view = array_view<real, 3>;
using double1d_view = array_view<real, 1>;
using double2d_view = array_view<real, 2>;
using double3d_view = array_view<real, 3>;
using const_real1d_view = array_view<const real, 1>;
using const_real2d_view = array_view<const real, 2>;
using const_real3d_view = array_view<const real, 3>;
using const_double1d_view = array_view<const double, 1>;
using const_double2d_view = array_view<const double, 2>;
using const_double3d_view = array_view<const double, 3>;

auto policy = std::execution::par;

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are initialized but remain static over the coure of the simulation
///////////////////////////////////////////////////////////////////////////////////////
struct Fixed_data {
  int nx, nz;                 //Number of local grid cells in the x- and z- dimensions for this MPI task
  int i_beg, k_beg;           //beginning index in the x- and z-directions for this MPI task
  int nranks, myrank;         //Number of MPI ranks and my rank id
  int left_rank, right_rank;  //MPI Rank IDs that exist to my left and right in the global domain
  int mainproc;             //Am I the main process (rank == 0)?
  real1d_container hy_dens_cell;        //hydrostatic density (vert cell avgs).   Dimensions: (1-hs:nz+hs)
  real1d_container hy_dens_theta_cell;  //hydrostatic rho*t (vert cell avgs).     Dimensions: (1-hs:nz+hs)
  real1d_container hy_dens_int;         //hydrostatic density (vert cell interf). Dimensions: (1:nz+1)
  real1d_container hy_dens_theta_int;   //hydrostatic rho*t (vert cell interf).   Dimensions: (1:nz+1)
  real1d_container hy_pressure_int;     //hydrostatic press (vert cell interf).   Dimensions: (1:nz+1)
};

//Declaring the functions defined after "main"
std::tuple<real3d_container, real3d_container, real3d_container, real3d_container, Fixed_data> init ( real &dt );
void finalize             ( );
void injection            ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht );
void density_current      ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht );
void gravity_waves        ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht );
void thermal              ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht );
void collision            ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht );
void hydro_const_theta    ( real z                    , real &r , real &t );
void hydro_const_bvfreq   ( real z , real bv_freq0    , real &r , real &t );
real sample_ellipse_cosine( real x , real z , real amp , real x0 , real z0 , real xrad , real zrad );
void output               ( const_real3d_view state , real etime , int &num_out , Fixed_data const &fixed_data );
void ncwrap               ( int ierr , int line );
void perform_timestep     ( real3d_view state , real3d_view state_tmp , real dt , real3d_view &flux , real3d_view &tend , int &direction_switch , Fixed_data const &fixed_data );
void semi_discrete_step   ( const_real3d_view const state_init , real3d_view const &state_forcing , real3d_view const &state_out , real dt , real3d_view flux , real3d_view tend , int dir , Fixed_data const &fixed_data );
void compute_tendencies_x ( const_real3d_view state , real3d_view const &flux , real3d_view const &tend , real dt , Fixed_data const &fixed_data );
void compute_tendencies_z ( const_real3d_view state , real3d_view const &flux , real3d_view const &tend , real dt , Fixed_data const &fixed_data );
void set_halo_values_x    ( real3d_view const &state  , Fixed_data const &fixed_data );
void set_halo_values_z    ( real3d_view const &state  , Fixed_data const &fixed_data );
void reductions           ( const_real3d_view state , double &mass , double &te , Fixed_data const &fixed_data );


///////////////////////////////////////////////////////////////////////////////////////
// THE MAIN PROGRAM STARTS HERE
///////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  MPI_Init(&argc,&argv);
  yakl::init();
  {
    real dt;                    //Model time step (seconds)

    // init allocates state
    auto [ state_container, state_tmp_container, flux_container, tend_container, fixed_data ] = init( dt ); // Should return fixed_data
    real3d_view state(state_container.data(),state_container.extents());
    real3d_view state_tmp(state_tmp_container.data(),state_tmp_container.extents());
    real3d_view flux(flux_container.data(), flux_container.extents());
    real3d_view tend(tend_container.data(),tend_container.extents());

    auto &mainproc = fixed_data.mainproc;

    //Initial reductions for mass, kinetic energy, and total energy
    double mass0, te0;
    reductions(state,mass0,te0,fixed_data);

    int  num_out = 0;          //The number of outputs performed so far
    real output_counter = 0;   //Helps determine when it's time to do output
    real etime = 0;

    //Output the initial state
    if (output_freq >= 0) {
      output(state,etime,num_out,fixed_data);
    }

    int direction_switch = 1;  // Tells dimensionally split which order to take x,z solves

    ////////////////////////////////////////////////////
    // MAIN TIME STEP LOOP
    ////////////////////////////////////////////////////
    auto t1 = std::chrono::steady_clock::now();
    while (etime < sim_time) {
      //If the time step leads to exceeding the simulation time, shorten it for the last step
      if (etime + dt > sim_time) { dt = sim_time - etime; }
      //Perform a single time step
      perform_timestep(state,state_tmp,dt,flux,tend,direction_switch,fixed_data);
      //Inform the user
      #ifndef NO_INFORM
        if (mainproc) { printf( "Elapsed Time: %lf / %lf\n", etime , sim_time ); }
      #endif
      //Update the elapsed time and output counter
      etime = etime + dt;
      output_counter = output_counter + dt;
      //If it's time for output, reset the counter, and do output
      if (output_freq >= 0 && output_counter >= output_freq) {
        output_counter = output_counter - output_freq;
        output(state,etime,num_out,fixed_data);
      }
    }
    auto t2 = std::chrono::steady_clock::now();
    if (mainproc) {
      std::cout << "CPU Time: " << std::chrono::duration<double>(t2-t1).count() << " sec\n";
    }

    //Final reductions for mass, kinetic energy, and total energy
    double mass, te;
    reductions(state,mass,te,fixed_data);

    if (mainproc) {
      printf( "d_mass: %le\n" , (mass - mass0)/mass0 );
      printf( "d_te:   %le\n" , (te   - te0  )/te0   );
    }

    finalize();
  }
  yakl::finalize();
  MPI_Finalize();
}


//Performs a single dimensionally split time step using a simple low-storate three-stage Runge-Kutta time integrator
//The dimensional splitting is a second-order-accurate alternating Strang splitting in which the
//order of directions is alternated each time step.
//The Runge-Kutta method used here is defined as follows:
// q*     = q_n + dt/3 * rhs(q_n)
// q**    = q_n + dt/2 * rhs(q* )
// q_n+1  = q_n + dt/1 * rhs(q**)
void perform_timestep( real3d_view state , real3d_view state_tmp , real dt , real3d_view &flux , real3d_view &tend , int &direction_switch , Fixed_data const &fixed_data ) {

  if (direction_switch) {
    //x-direction first
    semi_discrete_step( state , state     , state_tmp , dt / 3 , flux , tend , DIR_X , fixed_data );
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , flux , tend , DIR_X , fixed_data );
    semi_discrete_step( state , state_tmp , state     , dt / 1 , flux , tend , DIR_X , fixed_data );
    //z-direction second
    semi_discrete_step( state , state     , state_tmp , dt / 3 , flux , tend , DIR_Z , fixed_data );
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , flux , tend , DIR_Z , fixed_data );
    semi_discrete_step( state , state_tmp , state     , dt / 1 , flux , tend , DIR_Z , fixed_data );
  } else {
    //z-direction second
    semi_discrete_step( state , state     , state_tmp , dt / 3 , flux , tend , DIR_Z , fixed_data );
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , flux , tend , DIR_Z , fixed_data );
    semi_discrete_step( state , state_tmp , state     , dt / 1 , flux , tend , DIR_Z , fixed_data );
    //x-direction first
    semi_discrete_step( state , state     , state_tmp , dt / 3 , flux , tend , DIR_X , fixed_data );
    semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , flux , tend , DIR_X , fixed_data );
    semi_discrete_step( state , state_tmp , state     , dt / 1 , flux , tend , DIR_X , fixed_data );
  }
  if (direction_switch) { direction_switch = 0; } else { direction_switch = 1; }
}


//Perform a single semi-discretized step in time with the form:
//state_out = state_init + dt * rhs(state_forcing)
//Meaning the step starts from state_init, computes the rhs using state_forcing, and stores the result in state_out
void semi_discrete_step( const_real3d_view const state_init , real3d_view const &state_forcing , real3d_view const &state_out , real dt , real3d_view flux , real3d_view tend , int dir , Fixed_data const &fixed_data ) {
  auto &nx                 = fixed_data.nx                ;
  auto &nz                 = fixed_data.nz                ;
  auto &i_beg              = fixed_data.i_beg             ;
  auto &k_beg              = fixed_data.k_beg             ;
  auto &hy_dens_cell       = fixed_data.hy_dens_cell      ;

  if        (dir == DIR_X) {
    //Set the halo values for this MPI task's fluid state in the x-direction
    yakl::timer_start("halo x");
    set_halo_values_x(state_forcing,fixed_data);
    yakl::timer_stop("halo x");
    //Compute the time tendencies for the fluid state in the x-direction
    yakl::timer_start("tendencies x");
    compute_tendencies_x(state_forcing,flux,tend,dt,fixed_data);
    yakl::timer_stop("tendencies x");
  } else if (dir == DIR_Z) {
    //Set the halo values for this MPI task's fluid state in the z-direction
    yakl::timer_start("halo z");
    set_halo_values_z(state_forcing,fixed_data);
    yakl::timer_stop("halo z");
    //Compute the time tendencies for the fluid state in the z-direction
    yakl::timer_start("tendencies z");
    compute_tendencies_z(state_forcing,flux,tend,dt,fixed_data);
    yakl::timer_stop("tendencies z");
  }

  /////////////////////////////////////////////////
  // TODO: MAKE THESE 3 LOOPS A PARALLEL_FOR
  /////////////////////////////////////////////////
  //Apply the tendencies to the fluid state
  yakl::timer_start("apply tendencies");
  int _i_beg = i_beg,
      _k_beg = k_beg,
      _hs    = hs;
  double _dt = dt;
  auto range1 = std::views::iota(0,NUM_VARS*nz*nx);
  std::for_each(policy,range1.begin(),range1.end(),[=](int idx)
    {
      auto [i, k, ll] = idx3d(idx, nx, nz);
      //if (data_spec_int == DATA_SPEC_GRAVITY_WAVES) {
      //  real x = (_i_beg + i+0.5)*dx;
      //  real z = (_k_beg + k+0.5)*dz;
      //  real wpert = sample_ellipse_cosine( x,z , 0.01 , xlen/8,1000., 500.,500. );
      //  tend(ID_WMOM,k,i) += wpert*hy_dens_cell(_hs+k);
      //}
      state_out(ll,_hs+k,_hs+i) = state_init(ll,_hs+k,_hs+i) + _dt * tend(ll,k,i);
    });
  yakl::timer_stop("apply tendencies");
}


//Compute the time tendencies of the fluid state using forcing in the x-direction
//Since the halos are set in a separate routine, this will not require MPI
//First, compute the flux vector at each cell interface in the x-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
void compute_tendencies_x( const_real3d_view state , real3d_view const &flux, real3d_view const &tend , real dt , Fixed_data const &fixed_data ) {
  auto &nx                 = fixed_data.nx                ;
  auto &nz                 = fixed_data.nz                ;
  auto &hy_dens_cell       = fixed_data.hy_dens_cell      ;
  auto &hy_dens_theta_cell = fixed_data.hy_dens_theta_cell;

  //Compute the hyperviscosity coeficient
  real hv_coef = -hv_beta * dx / (16*dt);
  //Compute fluxes in the x-direction for each cell
  auto range1 = std::views::iota(0, (nx+1)*nz);
  std::for_each(policy, range1.begin(), range1.end(), [=](int idx)
  {
    auto [i, k] = idx2d(idx,(nx+1));
      SArray<real,1,4> stencil;
      SArray<real,1,NUM_VARS> d3_vals;
      SArray<real,1,NUM_VARS> vals;
      //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
      for (int ll=0; ll<NUM_VARS; ll++) {
        for (int s=0; s < sten_size; s++) {
          stencil(s) = state(ll,hs+k,i+s);
        }
        //Fourth-order-accurate interpolation of the state
        vals(ll) = -stencil(0)/12 + 7*stencil(1)/12 + 7*stencil(2)/12 - stencil(3)/12;
        //First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
        d3_vals(ll) = -stencil(0) + 3*stencil(1) - 3*stencil(2) + stencil(3);
      }

      //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
      real r = vals(ID_DENS) + hy_dens_cell(hs+k);
      real u = vals(ID_UMOM) / r;
      real w = vals(ID_WMOM) / r;
      real t = ( vals(ID_RHOT) + hy_dens_theta_cell(hs+k) ) / r;
      real p = C0*pow((r*t),gamm);

      //Compute the flux vector
      flux(ID_DENS,k,i) = r*u     - hv_coef*d3_vals(ID_DENS);
      flux(ID_UMOM,k,i) = r*u*u+p - hv_coef*d3_vals(ID_UMOM);
      flux(ID_WMOM,k,i) = r*u*w   - hv_coef*d3_vals(ID_WMOM);
      flux(ID_RHOT,k,i) = r*u*t   - hv_coef*d3_vals(ID_RHOT);
  });

  //Use the fluxes to compute tendencies for each cell
  auto range2 = std::views::iota(0, NUM_VARS*nz*nx);
  std::for_each(policy, range2.begin(), range2.end(), [=](int idx)
  {
    auto [ i, k, ll ] = idx3d(idx,nx,nz);
    tend(ll,k,i) = -( flux(ll,k,i+1) - flux(ll,k,i) ) / dx;
  });
}


//Compute the time tendencies of the fluid state using forcing in the z-direction
//Since the halos are set in a separate routine, this will not require MPI
//First, compute the flux vector at each cell interface in the z-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
void compute_tendencies_z( const_real3d_view state , real3d_view const &flux , real3d_view const &tend , real dt , Fixed_data const &fixed_data ) {
  auto &nx                 = fixed_data.nx                ;
  auto &nz                 = fixed_data.nz                ;
  auto &hy_dens_int        = fixed_data.hy_dens_int       ;
  auto &hy_dens_theta_int  = fixed_data.hy_dens_theta_int ;
  auto &hy_pressure_int    = fixed_data.hy_pressure_int   ;

  //Compute the hyperviscosity coeficient
  real hv_coef = -hv_beta * dz / (16*dt);
  //Compute fluxes in the x-direction for each cell
  auto range1 = std::views::iota(0, nx*(nz+1));
  std::for_each(policy, range1.begin(), range1.end(), [=](int idx)
  {
    auto [i, k] = idx2d(idx,nx);
      SArray<real,1,4> stencil;
      SArray<real,1,NUM_VARS> d3_vals;
      SArray<real,1,NUM_VARS> vals;
      //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
      for (int ll=0; ll<NUM_VARS; ll++) {
        for (int s=0; s<sten_size; s++) {
          stencil(s) = state(ll,k+s,hs+i);
        }
        //Fourth-order-accurate interpolation of the state
        vals(ll) = -stencil(0)/12 + 7*stencil(1)/12 + 7*stencil(2)/12 - stencil(3)/12;
        //First-order-accurate interpolation of the third spatial derivative of the state
        d3_vals(ll) = -stencil(0) + 3*stencil(1) - 3*stencil(2) + stencil(3);
      }

      //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
      real r = vals(ID_DENS) + hy_dens_int(k);
      real u = vals(ID_UMOM) / r;
      real w = vals(ID_WMOM) / r;
      real t = ( vals(ID_RHOT) + hy_dens_theta_int(k) ) / r;
      real p = C0*pow((r*t),gamm) - hy_pressure_int(k);
      if (k == 0 || k == nz) {
        w                = 0;
        d3_vals(ID_DENS) = 0;
      }

      //Compute the flux vector with hyperviscosity
      flux(ID_DENS,k,i) = r*w     - hv_coef*d3_vals(ID_DENS);
      flux(ID_UMOM,k,i) = r*w*u   - hv_coef*d3_vals(ID_UMOM);
      flux(ID_WMOM,k,i) = r*w*w+p - hv_coef*d3_vals(ID_WMOM);
      flux(ID_RHOT,k,i) = r*w*t   - hv_coef*d3_vals(ID_RHOT);
  });

  //Use the fluxes to compute tendencies for each cell
  auto range2 = std::views::iota(0,NUM_VARS*nz*nx);
  std::for_each(policy,range2.begin(),range2.end(),[=](int idx)
  {
    auto [ i, k, ll ] = idx3d(idx,nx,nz);
        tend(ll,k,i) = -( flux(ll,k+1,i) - flux(ll,k,i) ) / dz;
        if (ll == ID_WMOM) {
          tend(ll,k,i) -= state(ID_DENS,hs+k,hs+i)*grav;
        }
  });
}



//Set this MPI task's halo values in the x-direction. This routine will require MPI
void set_halo_values_x( real3d_view const &state , Fixed_data const &fixed_data ) {
  auto &nx                 = fixed_data.nx                ;
  auto &nz                 = fixed_data.nz                ;
  auto &k_beg              = fixed_data.k_beg             ;
  auto &left_rank          = fixed_data.left_rank         ;
  auto &right_rank         = fixed_data.right_rank        ;
  auto &myrank             = fixed_data.myrank            ;
  auto &hy_dens_cell       = fixed_data.hy_dens_cell      ;
  auto &hy_dens_theta_cell = fixed_data.hy_dens_theta_cell;

  ////////////////////////////////////////////////////////////////////////
  // TODO: EXCHANGE HALO VALUES WITH NEIGHBORING MPI TASKS
  // (1) give    state(1:hs,1:nz,1:NUM_VARS)       to   my left  neighbor
  // (2) receive state(1-hs:0,1:nz,1:NUM_VARS)     from my left  neighbor
  // (3) give    state(nx-hs+1:nx,1:nz,1:NUM_VARS) to   my right neighbor
  // (4) receive state(nx+1:nx+hs,1:nz,1:NUM_VARS) from my right neighbor
  ////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////
  // DELETE THE SERIAL CODE BELOW AND REPLACE WITH MPI
  //////////////////////////////////////////////////////
  auto range1 = std::views::iota(0, nz*NUM_VARS);
  std::for_each(policy, range1.begin(), range1.end(), [=](int idx)
  {
    auto [ k, ll ] = idx2d(idx,nz);
      state(ll,hs+k,0      ) = state(ll,hs+k,nx+hs-2);
      state(ll,hs+k,1      ) = state(ll,hs+k,nx+hs-1);
      state(ll,hs+k,nx+hs  ) = state(ll,hs+k,hs     );
      state(ll,hs+k,nx+hs+1) = state(ll,hs+k,hs+1   );
  });
  ////////////////////////////////////////////////////

  if (data_spec_int == DATA_SPEC_INJECTION) {
    if (myrank == 0) {
      auto range2 = std::views::iota(0, nz*hs);
      std::for_each(policy, range2.begin(), range2.end(), [=](int idx)
      {
        auto [ i, k ] = idx2d(idx,hs);
          real z = (k_beg + k+0.5)*dz;
          if (abs(z-3*zlen/4) <= zlen/16) {
            state(ID_UMOM,hs+k,i) = (state(ID_DENS,hs+k,i)+hy_dens_cell(hs+k)) * 50.;
            state(ID_RHOT,hs+k,i) = (state(ID_DENS,hs+k,i)+hy_dens_cell(hs+k)) * 298. - hy_dens_theta_cell(hs+k);
          }
      });
    }
  }
}


//Set this MPI task's halo values in the z-direction. This does not require MPI because there is no MPI
//decomposition in the vertical direction
void set_halo_values_z( real3d_view const &state , Fixed_data const &fixed_data ) {
  auto &nx                 = fixed_data.nx                ;
  auto &nz                 = fixed_data.nz                ;
  auto &hy_dens_cell       = fixed_data.hy_dens_cell      ;
  
  auto range1 = std::views::iota(0, (nx+2*hs)*NUM_VARS);
  std::for_each(policy, range1.begin(), range1.end(), [=](int idx)
  {
    auto [ i, ll ] = idx2d(idx,(nx+2*hs));
      if (ll == ID_WMOM) {
        state(ll,0      ,i) = 0.;
        state(ll,1      ,i) = 0.;
        state(ll,nz+hs  ,i) = 0.;
        state(ll,nz+hs+1,i) = 0.;
      } else if (ll == ID_UMOM) {
        state(ll,0      ,i) = state(ll,hs     ,i) / hy_dens_cell(hs     ) * hy_dens_cell(0      );
        state(ll,1      ,i) = state(ll,hs     ,i) / hy_dens_cell(hs     ) * hy_dens_cell(1      );
        state(ll,nz+hs  ,i) = state(ll,nz+hs-1,i) / hy_dens_cell(nz+hs-1) * hy_dens_cell(nz+hs  );
        state(ll,nz+hs+1,i) = state(ll,nz+hs-1,i) / hy_dens_cell(nz+hs-1) * hy_dens_cell(nz+hs+1);
      } else {
        state(ll,0      ,i) = state(ll,hs     ,i);
        state(ll,1      ,i) = state(ll,hs     ,i);
        state(ll,nz+hs  ,i) = state(ll,nz+hs-1,i);
        state(ll,nz+hs+1,i) = state(ll,nz+hs-1,i);
      }
  });
}


std::tuple<real3d_container, real3d_container, real3d_container, real3d_container, Fixed_data> init( real &dt ) {
  int nx;
  int nz;
  int i_beg;
  int k_beg;
  int left_rank;
  int right_rank;
  int nranks;
  int myrank;
  int mainproc;
  int ierr;

  /////////////////////////////////////////////////////////////
  // BEGIN MPI DUMMY SECTION
  // TODO: (1) GET NUMBER OF MPI RANKS
  //       (2) GET MY MPI RANK ID (RANKS ARE ZERO-BASED INDEX)
  //       (3) COMPUTE MY BEGINNING "I" INDEX (1-based index)
  //       (4) COMPUTE HOW MANY X-DIRECTION CELLS MY RANK HAS
  //       (5) FIND MY LEFT AND RIGHT NEIGHBORING RANK IDs
  /////////////////////////////////////////////////////////////
  nranks = 1;
  myrank = 0;
  i_beg = 0;
  nx = nx_glob;
  left_rank = 0;
  right_rank = 0;
  //////////////////////////////////////////////
  // END MPI DUMMY SECTION
  //////////////////////////////////////////////

  //Vertical direction isn't MPI-ized, so the rank's local values = the global values
  k_beg = 0;
  nz = nz_glob;
  mainproc = (myrank == 0);

  //Allocate the model data
  auto state              = real3d_container(NUM_VARS,nz+2*hs,nx+2*hs);

  //Define the maximum stable time step based on an assumed maximum wind speed
  dt = min(dx,dz) / max_speed * cfl;

  //If I'm the main process in MPI, display some grid information
  if (mainproc) {
    printf( "nx_glob, nz_glob: %d %d\n", nx_glob, nz_glob);
    printf( "dx,dz: %lf %lf\n",dx,dz);
    printf( "dt: %lf\n",dt);
  }
  //Want to make sure this info is displayed before further output
  ierr = MPI_Barrier(MPI_COMM_WORLD);

  // Define quadrature weights and points
  const int nqpoints = 3;
  SArray<real,1,nqpoints> qpoints;
  SArray<real,1,nqpoints> qweights;

  qpoints(0) = 0.112701665379258311482073460022;
  qpoints(1) = 0.500000000000000000000000000000;
  qpoints(2) = 0.887298334620741688517926539980;

  qweights(0) = 0.277777777777777777777777777779;
  qweights(1) = 0.444444444444444444444444444444;
  qweights(2) = 0.277777777777777777777777777779;

  //////////////////////////////////////////////////////////////////////////
  // Initialize the cell-averaged fluid state via Gauss-Legendre quadrature
  //////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////
  // TODO: MAKE THESE 2 LOOPS A PARALLEL_FOR
  /////////////////////////////////////////////////
  for (int k=0; k<nz+2*hs; k++) {
    for (int i=0; i<nx+2*hs; i++) {
      //Initialize the state to zero
      for (int ll=0; ll<NUM_VARS; ll++) {
        state(ll,k,i) = 0.;
      }
      //Use Gauss-Legendre quadrature to initialize a hydrostatic balance + temperature perturbation
      for (int kk=0; kk<nqpoints; kk++) {
        for (int ii=0; ii<nqpoints; ii++) {
          //Compute the x,z location within the global domain based on cell and quadrature index
          real x = (i_beg + i-hs+0.5)*dx + (qpoints(ii)-0.5)*dx;
          real z = (k_beg + k-hs+0.5)*dz + (qpoints(kk)-0.5)*dz;
          real r, u, w, t, hr, ht;

          //Set the fluid state based on the user's specification
          if (data_spec_int == DATA_SPEC_COLLISION      ) { collision      (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_THERMAL        ) { thermal        (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_GRAVITY_WAVES  ) { gravity_waves  (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) { density_current(x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_INJECTION      ) { injection      (x,z,r,u,w,t,hr,ht); }

          //Store into the fluid state array
          state(ID_DENS,k,i) += r                         * qweights(ii)*qweights(kk);
          state(ID_UMOM,k,i) += (r+hr)*u                  * qweights(ii)*qweights(kk);
          state(ID_WMOM,k,i) += (r+hr)*w                  * qweights(ii)*qweights(kk);
          state(ID_RHOT,k,i) += ( (r+hr)*(t+ht) - hr*ht ) * qweights(ii)*qweights(kk);
        }
      }
    }
  }

  auto hy_dens_cell       = real1d_container(nz+2*hs);
  auto hy_dens_theta_cell = real1d_container(nz+2*hs);
  auto hy_dens_int        = real1d_container(nz+1);
  auto hy_dens_theta_int  = real1d_container(nz+1);
  auto hy_pressure_int    = real1d_container(nz+1);

  //Compute the hydrostatic background state over vertical cell averages
  /////////////////////////////////////////////////
  // TODO: MAKE THIS LOOP A PARALLEL_FOR
  /////////////////////////////////////////////////
  for (int k=0; k<nz+2*hs; k++) {
    hy_dens_cell      (k) = 0.;
    hy_dens_theta_cell(k) = 0.;
    for (int kk=0; kk<nqpoints; kk++) {
      real z = (k_beg + k-hs+0.5)*dz;
      real r, u, w, t, hr, ht;
      //Set the fluid state based on the user's specification
      if (data_spec_int == DATA_SPEC_COLLISION      ) { collision      (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_THERMAL        ) { thermal        (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_GRAVITY_WAVES  ) { gravity_waves  (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) { density_current(0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_INJECTION      ) { injection      (0.,z,r,u,w,t,hr,ht); }
      hy_dens_cell      (k) = hy_dens_cell      (k) + hr    * qweights(kk);
      hy_dens_theta_cell(k) = hy_dens_theta_cell(k) + hr*ht * qweights(kk);
    }
  }

  //Compute the hydrostatic background state at vertical cell interfaces
  /////////////////////////////////////////////////
  // TODO: MAKE THIS LOOP A PARALLEL_FOR
  /////////////////////////////////////////////////
  for (int k=0; k<nz+1; k++) {
    real z = (k_beg + k)*dz;
    real r, u, w, t, hr, ht;
    if (data_spec_int == DATA_SPEC_COLLISION      ) { collision      (0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_THERMAL        ) { thermal        (0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_GRAVITY_WAVES  ) { gravity_waves  (0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) { density_current(0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_INJECTION      ) { injection      (0.,z,r,u,w,t,hr,ht); }
    hy_dens_int      (k) = hr;
    hy_dens_theta_int(k) = hr*ht;
    hy_pressure_int  (k) = C0*pow((hr*ht),gamm);
  }
  real3d_container state_tmp_container(NUM_VARS,nz+2*hs,nx+2*hs);
  real3d_container flux_container(NUM_VARS,nz+1,nx+1);
  real3d_container tend_container(NUM_VARS,nz,nx);
  return {state,state_tmp_container,flux_container,tend_container,{
    nx, nz,
    i_beg, k_beg,           //beginning index in the x- and z-directions for this MPI task
    nranks, myrank,         //Number of MPI ranks and my rank id
    left_rank, right_rank,  //MPI Rank IDs that exist to my left and right in the global domain
    mainproc,             //Am I the main process (rank == 0)?
    hy_dens_cell,        //hydrostatic density (vert cell avgs).   Dimensions: (1-hs:nz+hs)
    hy_dens_theta_cell,  //hydrostatic rho*t (vert cell avgs).     Dimensions: (1-hs:nz+hs)
    hy_dens_int,         //hydrostatic density (vert cell interf). Dimensions: (1:nz+1)
    hy_dens_theta_int,   //hydrostatic rho*t (vert cell interf).   Dimensions: (1:nz+1)
    hy_pressure_int,     //hydrostatic press (vert cell interf).   Dimensions: (1:nz+1)
  }};
}


//This test case is initially balanced but injects fast, cold air from the left boundary near the model top
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void injection( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht ) {
  hydro_const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
}


//Initialize a density current (falling cold thermal that propagates along the model bottom)
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void density_current( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht ) {
  hydro_const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
  t = t + sample_ellipse_cosine(x,z,-20. ,xlen/2,5000.,4000.,2000.);
}


//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void gravity_waves ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht ) {
  hydro_const_bvfreq(z,0.02,hr,ht);
  r = 0.;
  t = 0.;
  u = 15.;
  w = 0.;
}


//Rising thermal
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void thermal( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht ) {
  hydro_const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
  t = t + sample_ellipse_cosine(x,z, 3. ,xlen/2,2000.,2000.,2000.);
}


//Colliding thermals
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void collision( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht ) {
  hydro_const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
  t = t + sample_ellipse_cosine(x,z, 20.,xlen/2,2000.,2000.,2000.);
  t = t + sample_ellipse_cosine(x,z,-20.,xlen/2,8000.,2000.,2000.);
}


//Establish hydrstatic balance using constant potential temperature (thermally neutral atmosphere)
//z is the input coordinate
//r and t are the output background hydrostatic density and potential temperature
void hydro_const_theta( real z , real &r , real &t ) {
  const real theta0 = 300.;  //Background potential temperature
  const real exner0 = 1.;    //Surface-level Exner pressure
  //Establish hydrostatic balance first using Exner pressure
  t = theta0;                                  //Potential Temperature at z
  real exner = exner0 - grav * z / (cp * theta0);   //Exner pressure at z
  real p = p0 * pow(exner,(cp/rd));                 //Pressure at z
  real rt = pow((p / C0),(1. / gamm));             //rho*theta at z
  r = rt / t;                                  //Density at z
}


//Establish hydrstatic balance using constant Brunt-Vaisala frequency
//z is the input coordinate
//bv_freq0 is the constant Brunt-Vaisala frequency
//r and t are the output background hydrostatic density and potential temperature
void hydro_const_bvfreq( real z , real bv_freq0 , real &r , real &t ) {
  const real theta0 = 300.;  //Background potential temperature
  const real exner0 = 1.;    //Surface-level Exner pressure
  t = theta0 * exp( bv_freq0*bv_freq0 / grav * z );                                    //Pot temp at z
  real exner = exner0 - grav*grav / (cp * bv_freq0*bv_freq0) * (t - theta0) / (t * theta0); //Exner pressure at z
  real p = p0 * pow(exner,(cp/rd));                                                         //Pressure at z
  real rt = pow((p / C0),(1. / gamm));                                                  //rho*theta at z
  r = rt / t;                                                                          //Density at z
}


//Sample from an ellipse of a specified center, radius, and amplitude at a specified location
//x and z are input coordinates
//amp,x0,z0,xrad,zrad are input amplitude, center, and radius of the ellipse
real sample_ellipse_cosine( real x , real z , real amp , real x0 , real z0 , real xrad , real zrad ) {
  //Compute distance from bubble center
  real dist = sqrt( ((x-x0)/xrad)*((x-x0)/xrad) + ((z-z0)/zrad)*((z-z0)/zrad) ) * pi / 2.;
  //If the distance from bubble center is less than the radius, create a cos**2 profile
  if (dist <= pi / 2.) {
    return amp * pow(cos(dist),2.);
  } else {
    return 0.;
  }
}


//Output the fluid state (state) to a NetCDF file at a given elapsed model time (etime)
//The file I/O uses parallel-netcdf, the only external library required for this mini-app.
//If it's too cumbersome, you can comment the I/O out, but you'll miss out on some potentially cool graphics
void output( const_real3d_view state , real etime , int &num_out , Fixed_data const &fixed_data ) {
  auto &nx                 = fixed_data.nx                ;
  auto &nz                 = fixed_data.nz                ;
  auto &i_beg              = fixed_data.i_beg             ;
  auto &k_beg              = fixed_data.k_beg             ;
  auto &mainproc         = fixed_data.mainproc        ;
  auto &hy_dens_cell       = fixed_data.hy_dens_cell      ;
  auto &hy_dens_theta_cell = fixed_data.hy_dens_theta_cell;

  int ncid, t_dimid, x_dimid, z_dimid, dens_varid, uwnd_varid, wwnd_varid, theta_varid, t_varid, dimids[3];
  MPI_Offset st1[1], ct1[1], st3[3], ct3[3];
  //Temporary arrays to hold density, u-wind, w-wind, and potential temperature (theta)
  //Inform the user
  if (mainproc) { printf("*** OUTPUT ***\n"); }
  //Allocate some (big) temp arrays
  double2d_container dens ( nz,nx );
  double2d_container uwnd ( nz,nx );
  double2d_container wwnd ( nz,nx );
  double2d_container theta( nz,nx );

  //If the elapsed time is zero, create the file. Otherwise, open the file
  if (etime == 0) {
    //Create the file
    ncwrap( ncmpi_create( MPI_COMM_WORLD , "output.nc" , NC_CLOBBER , MPI_INFO_NULL , &ncid ) , __LINE__ );
    //Create the dimensions
    ncwrap( ncmpi_def_dim( ncid , "t" , (MPI_Offset) NC_UNLIMITED , &t_dimid ) , __LINE__ );
    ncwrap( ncmpi_def_dim( ncid , "x" , (MPI_Offset) nx_glob      , &x_dimid ) , __LINE__ );
    ncwrap( ncmpi_def_dim( ncid , "z" , (MPI_Offset) nz_glob      , &z_dimid ) , __LINE__ );
    //Create the variables
    dimids[0] = t_dimid;
    ncwrap( ncmpi_def_var( ncid , "t"     , NC_DOUBLE , 1 , dimids ,     &t_varid ) , __LINE__ );
    dimids[0] = t_dimid; dimids[1] = z_dimid; dimids[2] = x_dimid;
    ncwrap( ncmpi_def_var( ncid , "dens"  , NC_DOUBLE , 3 , dimids ,  &dens_varid ) , __LINE__ );
    ncwrap( ncmpi_def_var( ncid , "uwnd"  , NC_DOUBLE , 3 , dimids ,  &uwnd_varid ) , __LINE__ );
    ncwrap( ncmpi_def_var( ncid , "wwnd"  , NC_DOUBLE , 3 , dimids ,  &wwnd_varid ) , __LINE__ );
    ncwrap( ncmpi_def_var( ncid , "theta" , NC_DOUBLE , 3 , dimids , &theta_varid ) , __LINE__ );
    //End "define" mode
    ncwrap( ncmpi_enddef( ncid ) , __LINE__ );
  } else {
    //Open the file
    ncwrap( ncmpi_open( MPI_COMM_WORLD , "output.nc" , NC_WRITE , MPI_INFO_NULL , &ncid ) , __LINE__ );
    //Get the variable IDs
    ncwrap( ncmpi_inq_varid( ncid , "dens"  ,  &dens_varid ) , __LINE__ );
    ncwrap( ncmpi_inq_varid( ncid , "uwnd"  ,  &uwnd_varid ) , __LINE__ );
    ncwrap( ncmpi_inq_varid( ncid , "wwnd"  ,  &wwnd_varid ) , __LINE__ );
    ncwrap( ncmpi_inq_varid( ncid , "theta" , &theta_varid ) , __LINE__ );
    ncwrap( ncmpi_inq_varid( ncid , "t"     ,     &t_varid ) , __LINE__ );
  }

  //Store perturbed values in the temp arrays for output
  /////////////////////////////////////////////////
  // TODO: MAKE THESE 2 LOOPS A PARALLEL_FOR
  /////////////////////////////////////////////////
  for (int k=0; k<nz; k++) {
    for (int i=0; i<nx; i++) {
      dens (k,i) = state(ID_DENS,hs+k,hs+i);
      uwnd (k,i) = state(ID_UMOM,hs+k,hs+i) / ( hy_dens_cell(hs+k) + state(ID_DENS,hs+k,hs+i) );
      wwnd (k,i) = state(ID_WMOM,hs+k,hs+i) / ( hy_dens_cell(hs+k) + state(ID_DENS,hs+k,hs+i) );
      theta(k,i) = ( state(ID_RHOT,hs+k,hs+i) + hy_dens_theta_cell(hs+k) ) / ( hy_dens_cell(hs+k) + state(ID_DENS,hs+k,hs+i) ) - hy_dens_theta_cell(hs+k) / hy_dens_cell(hs+k);
    }
  }

  //Write the grid data to file with all the processes writing collectively
  st3[0] = num_out; st3[1] = k_beg; st3[2] = i_beg;
  ct3[0] = 1      ; ct3[1] = nz   ; ct3[2] = nx   ;
  ncwrap( ncmpi_put_vara_double_all( ncid ,  dens_varid , st3 , ct3 , dens.data()  ) , __LINE__ );
  ncwrap( ncmpi_put_vara_double_all( ncid ,  uwnd_varid , st3 , ct3 , uwnd.data()  ) , __LINE__ );
  ncwrap( ncmpi_put_vara_double_all( ncid ,  wwnd_varid , st3 , ct3 , wwnd.data()  ) , __LINE__ );
  ncwrap( ncmpi_put_vara_double_all( ncid , theta_varid , st3 , ct3 , theta.data() ) , __LINE__ );

  //Only the main process needs to write the elapsed time
  //Begin "independent" write mode
  ncwrap( ncmpi_begin_indep_data(ncid) , __LINE__ );
  //write elapsed time to file
  if (mainproc) {
    st1[0] = num_out;
    ct1[0] = 1;
    double etimearr[1];
    etimearr[0] = etime; ncwrap( ncmpi_put_vara_double( ncid , t_varid , st1 , ct1 , etimearr ) , __LINE__ );
  }
  //End "independent" write mode
  ncwrap( ncmpi_end_indep_data(ncid) , __LINE__ );

  //Close the file
  ncwrap( ncmpi_close(ncid) , __LINE__ );

  //Increment the number of outputs
  num_out = num_out + 1;
}


//Error reporting routine for the PNetCDF I/O
void ncwrap( int ierr , int line ) {
  if (ierr != NC_NOERR) {
    printf("NetCDF Error at line: %d\n", line);
    printf("%s\n",ncmpi_strerror(ierr));
    exit(-1);
  }
}


void finalize() {
}


//Compute reduced quantities for error checking without resorting to the "ncdiff" tool
void reductions( const_real3d_view state , double &mass , double &te , Fixed_data const &fixed_data ) {
  auto &nx                 = fixed_data.nx                ;
  auto &nz                 = fixed_data.nz                ;
  auto &hy_dens_cell       = fixed_data.hy_dens_cell      ;
  auto &hy_dens_theta_cell = fixed_data.hy_dens_theta_cell;

  mass = 0;
  te   = 0;
  auto range = std::views::iota(0,nz*nx);
  std::tie(mass, te) = std::transform_reduce( policy, range.begin(), range.end(), std::make_pair(0.0,0.0),
    [=] (auto a, auto b) { return std::make_pair(a.first + b.first, a.second + b.second); },
    [=] (int idx )
    {
      auto [i, k] = idx2d(idx, nx);
      double r  =   state(ID_DENS,hs+k,hs+i) + hy_dens_cell(hs+k);             // Density
      double u  =   state(ID_UMOM,hs+k,hs+i) / r;                              // U-wind
      double w  =   state(ID_WMOM,hs+k,hs+i) / r;                              // W-wind
      double th = ( state(ID_RHOT,hs+k,hs+i) + hy_dens_theta_cell(hs+k) ) / r; // Potential Temperature (theta)
      double p  = C0*pow(r*th,gamm);                               // Pressure
      double t  = th / pow(p0/p,rd/cp);                            // Temperature
      double ke = r*(u*u+w*w);                                     // Kinetic Energy
      double ie = r*cv*t;                                          // Internal Energy
      double mass = r        *dx*dz; // Accumulate domain mass
      double te   = (ke + ie)*dx*dz; // Accumulate domain total energy
      return std::make_pair(mass, te);
    });
  double glob[2], loc[2];
  loc[0] = mass;
  loc[1] = te;
  int ierr = MPI_Allreduce(loc,glob,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  mass = glob[0];
  te   = glob[1];
}


