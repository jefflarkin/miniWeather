#pragma once

#include "miniWeather_common.hpp"

//Set this MPI task's halo values in the x-direction.
template<class MemorySpace>
void set_halo_values_x(
  host_serial_execution_policy /* exec_policy */,
  view_3d state,
  const global_const_scalars& scalars,
  const global_const_arrays<MemorySpace>& arrays)
{
  const int nx = scalars.nx;
  const int nz = scalars.nz;

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
  for (int ll = 0; ll < NUM_VARS; ++ll) {
    for (int k = 0; k < nz; ++k) {
      state(ll, k+hs, 0) = state(ll, k+hs, nx+hs-2);
      state(ll, k+hs, 1) = state(ll, k+hs, nx+hs-1);
      state(ll, k+hs, nx+hs) = state(ll, k+hs, hs);
      state(ll, k+hs, nx+hs+1) = state(ll, k+hs, hs+1);
    }
  }
  ////////////////////////////////////////////////////

  if (data_spec_int == DATA_SPEC_INJECTION) {
    if (scalars.myrank == 0) {
      auto hy_dens_cell = arrays.hy_dens_cell();
      auto hy_dens_theta_cell = arrays.hy_dens_theta_cell();
      const int k_beg = scalars.k_beg;
      for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < hs; ++i) {
          const double z = (k_beg + k+0.5)*dz;
          if (fabs(z-3*zlen/4) <= zlen/16) {
            state(ID_UMOM, k+hs, i) = (state(ID_DENS, k+hs, i) + hy_dens_cell[k+hs]) * 50.0;
            state(ID_RHOT, k+hs, i) = (state(ID_DENS, k+hs, i) + hy_dens_cell[k+hs]) * 298.0 -
              hy_dens_theta_cell[k+hs];
          }
        }
      }
    }
  }
}

//Set this MPI task's halo values in the z-direction. This does not require MPI because there is no MPI
//decomposition in the vertical direction
template<class MemorySpace>
void set_halo_values_z(
  host_serial_execution_policy /* exec_policy */,
  view_3d state,
  const global_const_scalars& scalars,
  const global_const_arrays<MemorySpace>& arrays)
{
  const int nx = scalars.nx;
  const int nz = scalars.nz;
  auto hy_dens_cell = arrays.hy_dens_cell();

  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  for (int ll = 0; ll < NUM_VARS; ++ll) {
    for (int i = 0; i < nx+2*hs; ++i) {
      if (ll == ID_WMOM) {
        state(ll, 0, i) = 0.0;
        state(ll, 1, i) = 0.0;
        state(ll, nz+hs, i) = 0.0;
        state(ll, nz+hs+1, i) = 0.0;
      } else if (ll == ID_UMOM) {
        state(ll, 0, i) = state(ll, hs, i) / hy_dens_cell[hs] * hy_dens_cell[0];
        state(ll, 1, i) = state(ll, hs, i) / hy_dens_cell[hs] * hy_dens_cell[1];
        state(ll, nz+hs, i) = state(ll, nz+hs-1, i) / hy_dens_cell[nz+hs-1] * hy_dens_cell[nz+hs];
        state(ll, nz+hs+1, i) = state(ll, nz+hs-1, i) / hy_dens_cell[nz+hs-1] * hy_dens_cell[nz+hs+1];
      } else {
        state(ll, 0, i) = state(ll, hs, i);
        state(ll, 1, i) = state(ll, hs, i);
        state(ll, nz+hs, i) = state(ll, nz+hs-1, i);
        state(ll, nz+hs+1, i) = state(ll, nz+hs-1, i);
      }
    }
  }
}

//Compute the time tendencies of the fluid state using forcing in the x-direction
//Since the halos are set in a separate routine, this will not require MPI
//First, compute the flux vector at each cell interface in the x-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
template<class MemorySpace>
void compute_tendencies_x(
  host_serial_execution_policy /* exec_policy */,
  view_3d_const state,
  view_3d flux, view_3d tend, double dt,
  const global_const_scalars& scalars,
  const global_const_arrays<MemorySpace>& arrays)
{
  const int nx = scalars.nx;
  const int nz = scalars.nz;
  // Hyperviscosity coefficient
  const double hv_coef = -hv_beta * dx / (16*dt);
  auto hy_dens_cell = arrays.hy_dens_cell();
  auto hy_dens_theta_cell = arrays.hy_dens_theta_cell();

  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Compute fluxes in the x-direction for each cell
  for (int k = 0; k < nz; ++k) {
    for (int i = 0; i < nx+1; ++i) {
      //Use fourth-order interpolation from four cell averages
      //to compute the value at the interface in question
      std::array<double, NUM_VARS> d3_vals;
      std::array<double, NUM_VARS> vals;
      for (int ll = 0; ll < NUM_VARS; ++ll) {
        std::array<double, sten_size> stencil;
        for (int s = 0; s < sten_size; ++s) {
          stencil[s] = state(ll, k+hs, i+s);
        }
        //Fourth-order-accurate interpolation of the state
        vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
        //First-order-accurate interpolation of the third spatial derivative
        //of the state (for artificial viscosity)
        d3_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3];
      }

      //Compute density, u-wind, w-wind, potential temperature,
      //and pressure (r,u,w,t,p respectively)
      double r = vals[ID_DENS] + hy_dens_cell[k+hs];
      double u = vals[ID_UMOM] / r;
      double w = vals[ID_WMOM] / r;
      double t = ( vals[ID_RHOT] + hy_dens_theta_cell[k+hs] ) / r;
      double p = C0 * pow(r*t, gamm);

      //Compute the flux vector
      flux(ID_DENS, k, i) = r*u     - hv_coef*d3_vals[ID_DENS];
      flux(ID_UMOM, k, i) = r*u*u+p - hv_coef*d3_vals[ID_UMOM];
      flux(ID_WMOM, k, i) = r*u*w   - hv_coef*d3_vals[ID_WMOM];
      flux(ID_RHOT, k, i) = r*u*t   - hv_coef*d3_vals[ID_RHOT];
    }
  }

  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Use the fluxes to compute tendencies for each cell
  {
    view_3d_const flux_c = flux;
    for (int ll = 0; ll < NUM_VARS; ++ll) {
      for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < nx; ++i) {
          tend(ll, k, i) = -( flux_c(ll, k, i+1) - flux_c(ll, k, i) ) / dx;
        }
      }
    }
  }
}

//Compute the time tendencies of the fluid state using forcing in the z-direction
//Since the halos are set in a separate routine, this will not require MPI
//First, compute the flux vector at each cell interface in the z-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
template<class MemorySpace>
void compute_tendencies_z(
  host_serial_execution_policy /* exec_policy */,
  view_3d_const state,
  view_3d flux, view_3d tend, double dt,
  const global_const_scalars& scalars,
  const global_const_arrays<MemorySpace>& arrays)
{
  const int nx = scalars.nx;
  const int nz = scalars.nz;
  // Hyperviscosity coefficient
  const double hv_coef = -hv_beta * dz / (16*dt);
  auto hy_dens_int = arrays.hy_dens_int();
  auto hy_dens_theta_int = arrays.hy_dens_theta_int();
  auto hy_pressure_int = arrays.hy_pressure_int();

  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Compute fluxes in the x-direction for each cell
  for (int k = 0; k < nz+1; ++k) {
    for (int i = 0; i < nx; ++i) {
      //Use fourth-order interpolation from four cell averages
      //to compute the value at the interface in question
      std::array<double, NUM_VARS> d3_vals;
      std::array<double, NUM_VARS> vals;
      for (int ll = 0; ll < NUM_VARS; ++ll) {
        std::array<double, sten_size> stencil;
        for (int s = 0; s < sten_size; ++s) {
          stencil[s] = state(ll, k+s, i+hs);
        }
        //Fourth-order-accurate interpolation of the state
        vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
        //First-order-accurate interpolation of the third spatial derivative
        //of the state
        d3_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3];
      }

      //Compute density, u-wind, w-wind, potential temperature,
      //and pressure (r,u,w,t,p respectively)
      double r = vals[ID_DENS] + hy_dens_int[k];
      double u = vals[ID_UMOM] / r;
      double w = vals[ID_WMOM] / r;
      double t = (vals[ID_RHOT] + hy_dens_theta_int[k]) / r;
      double p = C0 * pow(r * t, gamm) - hy_pressure_int[k];
      //Enforce vertical boundary condition and exact mass conservation
      if (k == 0 || k == nz) {
        w                = 0;
        d3_vals[ID_DENS] = 0;
      }

      //Compute the flux vector with hyperviscosity
      flux(ID_DENS, k, i) = r*w     - hv_coef*d3_vals[ID_DENS];
      flux(ID_UMOM, k, i) = r*w*u   - hv_coef*d3_vals[ID_UMOM];
      flux(ID_WMOM, k, i) = r*w*w+p - hv_coef*d3_vals[ID_WMOM];
      flux(ID_RHOT, k, i) = r*w*t   - hv_coef*d3_vals[ID_RHOT];
    }
  }

  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Use the fluxes to compute tendencies for each cell
  {
    view_3d_const flux_c = flux;
    for (int ll = 0; ll < NUM_VARS; ++ll) {
      for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < nx; ++i) {
          tend(ll, k, i) = -( flux_c(ll, k+1, i) - flux_c(ll, k, i) ) / dz;
          if (ll == ID_WMOM) {
            tend(ll, k, i) = tend(ll, k, i) - state(ID_DENS, k+hs, i+hs)*grav;
          }
        }
      }
    }
  }
}

template<class MemorySpace>
void apply_tendencies_to_fluid_state(
  host_serial_execution_policy /* exec_policy */,
  view_3d_const state_init,
  view_3d state_out,
  double dt /* not scalars.dt */,
  view_3d tend,
  const global_const_scalars& scalars,
  const global_const_arrays<MemorySpace>& arrays)
{
  const int nx = scalars.nx;
  const int nz = scalars.nz;

  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////

  auto hy_dens_cell = arrays.hy_dens_cell();
  view_3d_const tend_c = tend;
  for (int ll = 0; ll < NUM_VARS; ++ll) {
    for (int k = 0; k < nz; ++k) {
      for (int i = 0; i < nx; ++i) {
        if (data_spec_int == DATA_SPEC_GRAVITY_WAVES) {
          const int i_beg = scalars.i_beg;
          const int k_beg = scalars.k_beg;
          const double x = (i_beg + i+0.5)*dx;
          const double z = (k_beg + k+0.5)*dz;
          const double wpert = sample_ellipse_cosine(x, z, 0.01, xlen/8, 1000.0, 500.0, 500.0);
          tend(ID_WMOM, k, i) += wpert * hy_dens_cell[hs+k];
        }
        state_out(ll, k+hs, i+hs) = state_init(ll, k+hs, i+hs) + dt * tend_c(ll, k, i);
      }
    }
  }
}

// Initialize the cell-averaged fluid state via Gauss-Legendre quadrature
void initialize_cell_averaged_fluid_state(
  host_serial_execution_policy /* exec_policy */,
  view_3d state, view_3d state_tmp,
  int nx, int nz,
  int i_beg, int k_beg)
{
  for (int k = 0; k < nz+2*hs; ++k) {
    for (int i = 0; i < nx+2*hs; ++i) {
      //Initialize the state to zero
      for (int ll = 0; ll < NUM_VARS; ++ll) {
        state(ll, k, i) = 0.0;
      }
      //Use Gauss-Legendre quadrature to initialize a hydrostatic balance + temperature perturbation
      for (int kk = 0; kk < nqpoints; ++kk) {
        for (int ii = 0; ii < nqpoints; ++ii) {
          //Compute the x,z location within the global domain based on cell and quadrature index
          const double x = (i_beg + i-hs+0.5)*dx + (qpoints[ii]-0.5)*dx;
          const double z = (k_beg + k-hs+0.5)*dz + (qpoints[kk]-0.5)*dz;

          //Set the fluid state based on the user's specification
          auto [r, u, w, t, hr, ht] = get_test_case(data_spec_int, x, z);

          //Store into the fluid state array
          state(ID_DENS, k, i) = state(ID_DENS, k, i) + r                         * qweights[ii]*qweights[kk];
          state(ID_UMOM, k, i) = state(ID_UMOM, k, i) + (r+hr)*u                  * qweights[ii]*qweights[kk];
          state(ID_WMOM, k, i) = state(ID_WMOM, k, i) + (r+hr)*w                  * qweights[ii]*qweights[kk];
          state(ID_RHOT, k, i) = state(ID_RHOT, k, i) + ( (r+hr)*(t+ht) - hr*ht ) * qweights[ii]*qweights[kk];
        }
      }
      for (int ll = 0; ll < NUM_VARS; ++ll) {
        state_tmp(ll, k, i) = state(ll, k, i);
      }
    }
  }
}

void compute_hydrostatic_background_state(
  host_serial_execution_policy /* exec_policy */,
  view_1d hy_dens_cell,
  view_1d hy_dens_theta_cell,
  view_1d hy_dens_int,
  view_1d hy_dens_theta_int,
  view_1d hy_pressure_int,
  int nz,
  int k_beg)
{
  //Compute the hydrostatic background state over vertical cell averages
  for (int k = 0; k < nz+2*hs; ++k) {
    hy_dens_cell[k] = 0.0;
    hy_dens_theta_cell[k] = 0.0;
    for (int kk = 0; kk < nqpoints; ++kk) {
      const double z = (k_beg + k-hs+0.5)*dz;
      //Set the fluid state based on the user's specification
      auto [r, u, w, t, hr, ht] = get_test_case(data_spec_int, 0.0, z);
      hy_dens_cell[k]       = hy_dens_cell[k]       + hr    * qweights[kk];
      hy_dens_theta_cell[k] = hy_dens_theta_cell[k] + hr*ht * qweights[kk];
    }
  }
  //Compute the hydrostatic background state at vertical cell interfaces
  for (int k = 0; k < nz+1; ++k) {
    const double z = (k_beg + k)*dz;
    auto [r, u, w, t, hr, ht] = get_test_case(data_spec_int, 0.0, z);
    hy_dens_int      [k] = hr;
    hy_dens_theta_int[k] = hr * ht;
    hy_pressure_int  [k] = C0 * pow(hr * ht, gamm);
  }
}

template<class MemorySpace>
reduction_result local_reductions(
  host_serial_execution_policy exec_policy,
  view_3d_const state,
  const global_const_scalars& const_scalars,
  const global_const_arrays<MemorySpace>& const_arrays)
{
  reduction_result result{0.0, 0.0};
  const int nx = const_scalars.nx;
  const int nz = const_scalars.nz;
  auto hy_dens_cell = const_arrays.hy_dens_cell();
  auto hy_dens_theta_cell = const_arrays.hy_dens_theta_cell();

  for (int k = 0; k < nz; ++k) {
    for (int i = 0; i < nx; ++i) {
      double r  =  state(ID_DENS, k+hs, i+hs) + hy_dens_cell[hs+k]; // Density
      double u  =  state(ID_UMOM, k+hs, i+hs) / r;                  // U-wind
      double w  =  state(ID_WMOM, k+hs, i+hs) / r;                  // W-wind
      double th = (state(ID_RHOT, k+hs, i+hs) + hy_dens_theta_cell[hs+k]) / r; // Potential Temperature (theta)
      double p  = C0 * pow(r * th, gamm);                           // Pressure
      double t  = th / pow(p0 / p, rd / cp);                        // Temperature
      double ke = r*(u*u+w*w);                                      // Kinetic Energy
      double ie = r*cv*t;                                           // Internal Energy
      result.mass += r        *dx*dz; // Accumulate domain mass
      result.te   += (ke + ie)*dx*dz; // Accumulate domain total energy
    }
  }
  return result;
}

