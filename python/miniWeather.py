
# //////////////////////////////////////////////////////////////////////////////////////////
# // miniWeather
# // Author: Matt Norman <normanmr@ornl.gov>  , Oak Ridge National Laboratory
# // This code simulates dry, stratified, compressible, non-hydrostatic fluid flows
# // For documentation, please see the attached documentation in the "documentation" folder
# //
# //////////////////////////////////////////////////////////////////////////////////////////

import sys
import timeit
import numpy as np
#include "pnetcdf.h"

# "real" in the original C++ code could be either float or double.
real = np.float64 # or np.float32

#
# Constants (ported from cpp/const.h)
#

hs: int = 2

# We don't like code that redefines pi,
# but we keep it just for now, to test that
# the Python gives the same results as the C++.
pi: real   = 3.14159265358979323846264338327
grav: real = 9.8     # Gravitational acceleration (m / s^2)
cp: real   = 1004.0  # Specific heat of dry air at constant pressure
cv: real   = 717.0   # Specific heat of dry air at constant volume
rd: real   = 287.0   # Dry air constant for equation of state (P=rho*rd*T)
p0: real   = 1.e5    # Standard pressure at the surface in Pascals
C0: real   = 27.5629410929725921310572974482 # Constant to translate potential temperature into pressure (P=C0*(rho*theta)**gamma)
# gamma=cp/Rd, have to call this gamm because "gamma" is taken (I hate C so much)
gamm: real = 1.40027894002789400278940027894

#
# Domain and stability-related constants
#

xlen: real = 2.e4    # Length of the domain in the x-direction (meters)
zlen: real = 1.e4    # Length of the domain in the z-direction (meters)
hv_beta: real = 0.05 # How strong to diffuse the solution: hv_beta \in [0:1]
cfl: real = 1.50     # "Courant, Friedrichs, Lewy" number (for numerical stability)
max_speed: real = 450 # Assumed maximum wave speed during the simulation (speed of sound + speed of wind) (meter / sec)
# "Halo" size: number of cells beyond the MPI tasks's domain
# needed for a full "stencil" of information for reconstruction
hs: int = 2
# Size of the stencil used for interpolation
sten_size: int = 4

# ///////////////////////////////////////////////////////////////////////////////////////
# // BEGIN USER-CONFIGURABLE PARAMETERS
# ///////////////////////////////////////////////////////////////////////////////////////
# The x-direction length is twice as long as the z-direction length
# So, you'll want to have nx_glob be twice as large as nz_glob
nx_glob: int = _NX              # Number of total cells in the x-direction
nz_glob: int = _NZ              # Number of total cells in the z-direction
sim_time: real = _SIM_TIME      # How many seconds to run the simulation
output_freq: real = _OUT_FREQ   # How frequently to output data to file (in seconds)
data_spec_int: int = _DATA_SPEC # How to initialize the data
# ///////////////////////////////////////////////////////////////////////////////////////
# // END USER-CONFIGURABLE PARAMETERS
# ///////////////////////////////////////////////////////////////////////////////////////
dx: real = xlen / nx_glob
dz: real = zlen / nz_glob


#
# Parameters for indexing and flags
#

NUM_VARS: int = 4           # Number of fluid state variables
ID_DENS: int = 0           #index for density ("rho")
ID_UMOM: int = 1           #index for momentum in the x-direction ("rho * u")
ID_WMOM: int = 2           #index for momentum in the z-direction ("rho * w")
ID_RHOT: int = 3           #index for density * potential temperature ("rho * theta")
DIR_X: int = 1              #Integer constant to express that this operation is in the x-direction
DIR_Z: int = 2              #Integer constant to express that this operation is in the z-direction
DATA_SPEC_COLLISION: int = 1
DATA_SPEC_THERMAL: int = 2
DATA_SPEC_GRAVITY_WAVES: int = 3
DATA_SPEC_DENSITY_CURRENT: int = 5
DATA_SPEC_INJECTION: int = 6

#
# These functions aid in porting from the original C++.
#

#typedef yakl::Array<real  ,1,yakl::memHost> real1d;
def real1d(name: string, nx: int):
  return np.zeros(nx, dtype=real)

#typedef yakl::Array<real  ,2,yakl::memHost> real2d;
def real2d(name: string, nx: int, nz: int):
  return np.zeros((nz, nx), dtype=real)

#typedef yakl::Array<real  ,3,yakl::memHost> real3d;
def real3d(name: string, nx: int, nz: int, nvars: int):
  return np.zeros((nvars, nz, nx), dtype=real)

#typedef yakl::Array<double,2,yakl::memHost> doub2d;
def doub2d(name: string, nx: int, nz: int):
  return np.zeros((nz, nx), dtype=real)

#typedef yakl::Array<real   const,1,yakl::memHost> realConst1d;
def realConst1d(name: string, nx: int):
  return np.zeros(nx, dtype=real)

#typedef yakl::Array<real   const,2,yakl::memHost> realConst2d;
def realConst2d(name: string, nx: int, nz: int):
  return np.zeros((nz, nx), dtype=real)

# ///////////////////////////////////////////////////////////////////////////////////////
# // Variables that are initialized but remain static over the course of the simulation
# ///////////////////////////////////////////////////////////////////////////////////////

class Fixed_data:
  def __init__(self):
    self.nx = 0          # Number of local grid cells in the x dimension for this MPI task
    self.nz = 0          # Number of local grid cells in the z dimension for this MPI task
    self.i_beg = 0       # beginning index in the x direction for this MPI task
    self.k_beg = 0       # beginning index in the z direction for this MPI task
    self.nranks = 0      # Number of MPI ranks
    self.myrank = 0      # My rank id
    self.left_rank = 0   # MPI Rank ID that exists to my left in the global domain
    self.right_rank = 0  # MPI Rank ID that exists to my right in the global domain
    self.mainproc = True # Am I the main process (rank == 0)?
    self.hy_dens_cell = None       # realConst1d: hydrostatic density (vert cell avgs).   Dimensions: (1-hs:nz+hs)
    self.hy_dens_theta_cell = None # realConst1d: hydrostatic rho*t (vert cell avgs).     Dimensions: (1-hs:nz+hs)
    self.hy_dens_int = None        # realConst1d: hydrostatic density (vert cell interf). Dimensions: (1:nz+1)
    self.hy_dens_theta_int = None  # realConst1d: hydrostatic rho*t (vert cell interf).   Dimensions: (1:nz+1)
    self.hy_pressure_int = None    # realConst1d: hydrostatic press (vert cell interf).   Dimensions: (1:nz+1)


# ///////////////////////////////////////////////////////////////////////////////////////
# // THE MAIN PROGRAM STARTS HERE
# ///////////////////////////////////////////////////////////////////////////////////////
def main() -> None:
  #MPI_Init(&argc,&argv);
  #yakl::init();

  # fixed_data: Fixed_data
  # state: real3d
  # dt: real: Model time step (seconds)
  (fixed_data, state, dt) = init()

  mainproc = fixed_data.mainproc

  # Initial reductions for mass, kinetic energy, and total energy
  (mass0, te0) = reductions(state, fixed_data)

  num_out: int = 0    # The number of outputs performed so far
  etime: real  = 0.0  # Elapsed time

  # Output the initial state
  if output_freq >= 0:
    num_out = output(state, etime, num_out, fixed_data)

  # ////////////////////////////////////////////////////
  # MAIN TIME STEP LOOP
  # ////////////////////////////////////////////////////
  def run_simulation():
    direction_switch: int = 1  # Order in which dimensional splitting takes x,z solves
    output_counter: real = 0.0 # Helps determine when it's time to do output

    while etime < sim_time:
      # If the time step leads to exceeding the simulation time, shorten it for the last step
      if etime + dt > sim_time:
        dt = sim_time - etime

      # Perform a single time step
      direction_switch = perform_timestep(state, dt, direction_switch, fixed_data)
      # Inform the user
      if mainproc:
        print(f"Elapsed Time: {etime}, Simulation Time: {sim_time}\n")
      # Update the elapsed time and output counter
      etime = etime + dt
      output_counter = output_counter + dt
      # If it's time for output, reset the counter, and do output
      if output_freq >= 0 and output_counter >= output_freq:
        output_counter = output_counter - output_freq
      num_out = output(state, etime, num_out, fixed_data)

  time_in_s = timeit.timeit(run_simulation(), number=1)
  if mainproc:
    print(f"CPU Time: {time_in_s} s\n")

  # Final reductions for mass, kinetic energy, and total energy
  (mass, te) = reductions(state, fixed_data)

  if mainproc:
    print(f"d_mass: {((mass - mass0)/mass0)}")
    print(f"d_te:   {((te   - te0  )/te0  )}")

  finalize()

  # yakl::finalize();
  # MPI_Finalize();


# Performs a single dimensionally split time step using a simple low-storage three-stage Runge-Kutta time integrator
# The dimensional splitting is a second-order-accurate alternating Strang splitting in which the
# order of directions is alternated each time step.
# The Runge-Kutta method used here is defined as follows:
#   q*     = q_n + dt/3 * rhs(q_n)
#   q**    = q_n + dt/2 * rhs(q* )
#   q_n+1  = q_n + dt/1 * rhs(q**)
def perform_timestep(
    state, # real3d const&, input parameter
    dt, # real, must be an input parameter
    direction_switch, # int&, in/out parameter, transformed to input and return value
    fixed_data # Fixed_data const &, input parameter
  ) -> int: # was void; now returns direction_switch

  nx = fixed_data.nx
  nz = fixed_data.nz

  state_tmp = real3d("state_tmp", NUM_VARS, nz+2*hs, nx+2*hs)

  if direction_switch != 0:
    # x-direction first
    semi_discrete_step(state, state    , state_tmp, dt / 3, DIR_X, fixed_data)
    semi_discrete_step(state, state_tmp, state_tmp, dt / 2, DIR_X, fixed_data)
    semi_discrete_step(state, state_tmp, state    , dt / 1, DIR_X, fixed_data)
    # z-direction second
    semi_discrete_step(state, state    , state_tmp, dt / 3, DIR_Z, fixed_data)
    semi_discrete_step(state, state_tmp, state_tmp, dt / 2, DIR_Z, fixed_data)
    semi_discrete_step(state, state_tmp, state    , dt / 1, DIR_Z, fixed_data)
  else:
    # z-direction second
    semi_discrete_step(state, state    , state_tmp, dt / 3, DIR_Z, fixed_data)
    semi_discrete_step(state, state_tmp, state_tmp, dt / 2, DIR_Z, fixed_data)
    semi_discrete_step(state, state_tmp, state    , dt / 1, DIR_Z, fixed_data)
    # x-direction first
    semi_discrete_step(state, state    , state_tmp, dt / 3, DIR_X, fixed_data)
    semi_discrete_step(state, state_tmp, state_tmp, dt / 2, DIR_X, fixed_data)
    semi_discrete_step(state, state_tmp, state    , dt / 1, DIR_X, fixed_data)

  if direction_switch:
    direction_switch = 0
  else:
    direction_switch = 1

  return direction_switch



# Perform a single semi-discretized step in time with the form:
# state_out = state_init + dt * rhs(state_forcing)
# Meaning the step starts from state_init, computes the rhs using state_forcing, and stores the result in state_out
def semi_discrete_step(
    state_init, # realConst3d
    state_forcing, # real3d const&
    state_out, # real3d const&,
    dt: real,
    dir: int,
    fixed_data # Fixed_data const&
  ) -> None:

  nx                 = fixed_data.nx
  nz                 = fixed_data.nz
  i_beg              = fixed_data.i_beg
  k_beg              = fixed_data.k_beg
  hy_dens_cell       = fixed_data.hy_dens_cell

  tend = real3d("tend", NUM_VARS, nz, nx)

  if dir == DIR_X:
    # Set the halo values for this MPI task's fluid state in the x-direction
    #yakl::timer_start("halo x");
    set_halo_values_x(state_forcing, fixed_data)
    #yakl::timer_stop("halo x");
    # Compute the time tendencies for the fluid state in the x-direction
    #yakl::timer_start("tendencies x");
    compute_tendencies_x(state_forcing, tend, dt, fixed_data)
    #yakl::timer_stop("tendencies x");
  elif dir == DIR_Z:
    # Set the halo values for this MPI task's fluid state in the z-direction
    #yakl::timer_start("halo z");
    set_halo_values_z(state_forcing, fixed_data)
    #yakl::timer_stop("halo z");
    # Compute the time tendencies for the fluid state in the z-direction
    #yakl::timer_start("tendencies z");
    compute_tendencies_z(state_forcing, tend, dt, fixed_data)
    #yakl::timer_stop("tendencies z");

  # /////////////////////////////////////////////////
  # // TODO: MAKE THESE 3 LOOPS A PARALLEL_FOR
  # /////////////////////////////////////////////////
  # Apply the tendencies to the fluid state
  # yakl::timer_start("apply tendencies");
  for ll in range(NUM_VARS):
    for k in range(nz):
      for i in range(nx):
        if data_spec_int == DATA_SPEC_GRAVITY_WAVES:
          x: real = (i_beg + i+0.5)*dx;
          z: real = (k_beg + k+0.5)*dz;
          wpert: real = sample_ellipse_cosine(x, z, 0.01, xlen/8, 1000.0, 500.0, 500.0)
          tend[ID_WMOM,k,i] += wpert*hy_dens_cell[hs+k]

        state_out(ll,hs+k,hs+i) = state_init(ll,hs+k,hs+i) + dt * tend(ll,k,i);

  # yakl::timer_stop("apply tendencies");

  # NOTE It's OK for this not to return anything,
  # as long as we can treat state_out as an output parameter.


# Compute the time tendencies of the fluid state using forcing in the x-direction
# Since the halos are set in a separate routine, this will not require MPI
# First, compute the flux vector at each cell interface in the x-direction (including hyperviscosity)
# Then, compute the tendencies using those fluxes
def compute_tendencies_x(
    state, # realConst3d
    tend, # real3d const&
    dt, # real
    fixed_data # Fixed_data const&
  ) -> None:

  nx                 = fixed_data.nx
  nz                 = fixed_data.nz
  hy_dens_cell       = fixed_data.hy_dens_cell
  hy_dens_theta_cell = fixed_data.hy_dens_theta_cell

  flux = real3d("flux", NUM_VARS, nz, nx+1)

  # Compute the hyperviscosity coefficient
  hv_coef: real = -hv_beta * dx / (16*dt)
  # /////////////////////////////////////////////////
  # TODO: MAKE THESE 2 LOOPS A PARALLEL_FOR
  # /////////////////////////////////////////////////
  # Compute fluxes in the x-direction for each cell
  for k in range(nz):
    for i in range(nx+1):

      # "Stack Array" -- local multidimensional array type with compile-time extents
      #SArray<real,1,4> stencil;
      #SArray<real,1,NUM_VARS> d3_vals;
      #SArray<real,1,NUM_VARS> vals;

      stencil = np.zeros((1, 4), dtype=real)
      d3_vals = np.zeros((1, NUM_VARS), dtype=real)
      vals = np.zeros((1, NUM_VARS), dtype=real)
      
      # Use fourth-order interpolation from four cell averages to compute the value at the interface in question
      for ll in range(NUM_VARS):
        for s in range(sten_size):
          stencil[s] = state[ll,hs+k,i+s]

        # Fourth-order-accurate interpolation of the state
        vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12
        # First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
        d3_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3]

      # Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
      r = vals[ID_DENS] + hy_dens_cell[hs+k]
      u = vals[ID_UMOM] / r
      w = vals[ID_WMOM] / r
      t = (vals[ID_RHOT] + hy_dens_theta_cell[hs+k]) / r
      p = C0*pow((r*t), gamm)

      # Compute the flux vector
      flux[ID_DENS,k,i] = r*u     - hv_coef*d3_vals[ID_DENS]
      flux[ID_UMOM,k,i] = r*u*u+p - hv_coef*d3_vals[ID_UMOM]
      flux[ID_WMOM,k,i] = r*u*w   - hv_coef*d3_vals[ID_WMOM]
      flux[ID_RHOT,k,i] = r*u*t   - hv_coef*d3_vals[ID_RHOT]

  # /////////////////////////////////////////////////
  # // TODO: MAKE THESE 3 LOOPS A PARALLEL_FOR
  # /////////////////////////////////////////////////
  # Use the fluxes to compute tendencies for each cell
  for ll in range(NUM_VARS):
    for k in range(nz):
      for i in range(nx):
        tend[ll,k,i] = -( flux[ll,k,i+1] - flux[ll,k,i] ) / dx

  # NOTE It's OK for this not to return anything,
  # as long as we can treat tend as an output parameter.


# Compute the time tendencies of the fluid state using forcing in the z-direction
# Since the halos are set in a separate routine, this will not require MPI
# First, compute the flux vector at each cell interface in the z-direction (including hyperviscosity)
# Then, compute the tendencies using those fluxes
def compute_tendencies_z(
    state, # realConst3d
    tend, # real3d const&
    dt, # real
    fixed_data # Fixed_data const&
  ) -> None:

  nx                 = fixed_data.nx
  nz                 = fixed_data.nz
  hy_dens_int        = fixed_data.hy_dens_int
  hy_dens_theta_int  = fixed_data.hy_dens_theta_int
  hy_pressure_int    = fixed_data.hy_pressure_int

  flux = real3d("flux", NUM_VARS, nz+1, nx)

  # Compute the hyperviscosity coefficient
  hv_coef: real = -hv_beta * dz / (16*dt);
  # /////////////////////////////////////////////////
  # TODO: MAKE THESE 2 LOOPS A PARALLEL_FOR
  # /////////////////////////////////////////////////
  # Compute fluxes in the x-direction for each cell
  for k in range(nz+1):
    for i in range(nx):
      # "Stack Array" -- local multidimensional array type with compile-time extents
      #SArray<real,1,4> stencil;
      #SArray<real,1,NUM_VARS> d3_vals;
      #SArray<real,1,NUM_VARS> vals;

      stencil = np.zeros((1, 4), dtype=real)
      d3_vals = np.zeros((1, NUM_VARS), dtype=real)
      vals = np.zeros((1, NUM_VARS), dtype=real)

      # Use fourth-order interpolation from four cell averages to compute the value at the interface in question
      for ll in range(NUM_VARS):
        for s in range(sten_size):
          stencil[s] = state[ll,k+s,hs+i];

        # Fourth-order-accurate interpolation of the state
        vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12
        # First-order-accurate interpolation of the third spatial derivative of the state
        d3_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3]

      # Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
      r: real = vals[ID_DENS] + hy_dens_int[k];
      u: real = vals[ID_UMOM] / r;
      w: real = vals[ID_WMOM] / r;
      t: real = ( vals[ID_RHOT] + hy_dens_theta_int[k] ) / r;
      p: real = C0*pow((r*t),gamm) - hy_pressure_int[k];
      if k == 0 or k == nz:
        w                = 0;
        d3_vals[ID_DENS] = 0;

      # Compute the flux vector with hyperviscosity
      flux[ID_DENS,k,i] = r*w     - hv_coef*d3_vals[ID_DENS];
      flux[ID_UMOM,k,i] = r*w*u   - hv_coef*d3_vals[ID_UMOM];
      flux[ID_WMOM,k,i] = r*w*w+p - hv_coef*d3_vals[ID_WMOM];
      flux[ID_RHOT,k,i] = r*w*t   - hv_coef*d3_vals[ID_RHOT];
    }
  }

  # Use the fluxes to compute tendencies for each cell
  #/////////////////////////////////////////////////
  # TODO: MAKE THESE 3 LOOPS A PARALLEL_FOR
  #/////////////////////////////////////////////////
  for ll in range(NUM_VARS):
    for k in range(nz):
      for i in range(nx):
        tend[ll,k,i] = -( flux[ll,k+1,i] - flux[ll,k,i] ) / dz
        if ll == ID_WMOM:
          tend[ll,k,i] -= state[ID_DENS,hs+k,hs+i]*grav

  # NOTE (mfh 2025/03/03) Don't need to return anything, as long as tend can be an output parameter


# Set this MPI task's halo values in the x-direction. This routine will require MPI
def set_halo_values_x(
    state, # real3d const&
    fixed_data # Fixed_data const&
  ) -> None:

  nx                 = fixed_data.nx
  nz                 = fixed_data.nz
  k_beg              = fixed_data.k_beg
  left_rank          = fixed_data.left_rank
  right_rank         = fixed_data.right_rank
  myrank             = fixed_data.myrank
  hy_dens_cell       = fixed_data.hy_dens_cell
  hy_dens_theta_cell = fixed_data.hy_dens_theta_cell

  # //////////////////////////////////////////////////////////////////////
  # TODO: EXCHANGE HALO VALUES WITH NEIGHBORING MPI TASKS
  # (1) give    state(1:hs,1:nz,1:NUM_VARS)       to   my left  neighbor
  # (2) receive state(1-hs:0,1:nz,1:NUM_VARS)     from my left  neighbor
  # (3) give    state(nx-hs+1:nx,1:nz,1:NUM_VARS) to   my right neighbor
  # (4) receive state(nx+1:nx+hs,1:nz,1:NUM_VARS) from my right neighbor
  # //////////////////////////////////////////////////////////////////////

  # //////////////////////////////////////////////////////
  # // DELETE THE SERIAL CODE BELOW AND REPLACE WITH MPI
  # //////////////////////////////////////////////////////
  for ll in range(NUM_VARS):
    for k in range(nz):
      state[ll,hs+k,0      ] = state[ll,hs+k,nx+hs-2];
      state[ll,hs+k,1      ] = state[ll,hs+k,nx+hs-1];
      state[ll,hs+k,nx+hs  ] = state[ll,hs+k,hs     ];
      state[ll,hs+k,nx+hs+1] = state[ll,hs+k,hs+1   ];

  if data_spec_int == DATA_SPEC_INJECTION:
    if myrank == 0:
      #/////////////////////////////////////////////////
      # TODO: MAKE THESE 2 LOOPS A PARALLEL_FOR
      #/////////////////////////////////////////////////
      for k in range(nz):
        for i in range(hs):
          z: real = (k_beg + k+0.5)*dz
          if abs(z-3*zlen/4) <= zlen/16:
            state[ID_UMOM,hs+k,i] = (state[ID_DENS,hs+k,i] + hy_dens_cell[hs+k]) * 50.0
            state[ID_RHOT,hs+k,i] = (state[ID_DENS,hs+k,i] + hy_dens_cell[hs+k]) * 298.0 - hy_dens_theta_cell[hs+k]

  # NOTE (mfh 2025/03/03) Don't need to return anything, as long as state can be an output parameter


# Set this MPI task's halo values in the z-direction. This does not require MPI because there is no MPI
# decomposition in the vertical direction
def set_halo_values_z(
    state, # real3d const&
    fixed_data # Fixed_data const&
  ) -> None:

  nx                 = fixed_data.nx
  nz                 = fixed_data.nz
  hy_dens_cell       = fixed_data.hy_dens_cell
 
  # /////////////////////////////////////////////////
  # // TODO: MAKE THESE 2 LOOPS A PARALLEL_FOR
  # /////////////////////////////////////////////////
  for ll in range(NUM_VARS):
    for i in range(nx+2*hs):
      if ll == ID_WMOM:
        state[ll,0      ,i] = 0.0
        state[ll,1      ,i] = 0.0
        state[ll,nz+hs  ,i] = 0.0
        state[ll,nz+hs+1,i] = 0.0
      elif ll == ID_UMOM:
        state[ll,0      ,i] = state[ll,hs     ,i] / hy_dens_cell[hs     ] * hy_dens_cell[0      ]
        state[ll,1      ,i] = state[ll,hs     ,i] / hy_dens_cell[hs     ] * hy_dens_cell[1      ]
        state[ll,nz+hs  ,i] = state[ll,nz+hs-1,i] / hy_dens_cell[nz+hs-1] * hy_dens_cell[nz+hs  ]
        state[ll,nz+hs+1,i] = state[ll,nz+hs-1,i] / hy_dens_cell[nz+hs-1] * hy_dens_cell[nz+hs+1]
      else:
        state[ll,0      ,i] = state[ll,hs     ,i]
        state[ll,1      ,i] = state[ll,hs     ,i]
        state[ll,nz+hs  ,i] = state[ll,nz+hs-1,i]
        state[ll,nz+hs+1,i] = state[ll,nz+hs-1,i]

  # NOTE (mfh 2025/03/03) Don't need to return anything, as long as state can be an output parameter


# state, dt, and fixed_data used to be output parameters.
# It would be more Pythonic to return them as a tuple.
def init(): # -> tuple[real3d, real, Fixed_data]:

  ierr: int = 0

  # /////////////////////////////////////////////////////////////
  #// BEGIN MPI DUMMY SECTION
  # // TODO: (1) GET NUMBER OF MPI RANKS
  # //       (2) GET MY MPI RANK ID (RANKS ARE ZERO-BASED INDEX)
  # //       (3) COMPUTE MY BEGINNING "I" INDEX (1-based index)
  # //       (4) COMPUTE HOW MANY X-DIRECTION CELLS MY RANK HAS
  # //       (5) FIND MY LEFT AND RIGHT NEIGHBORING RANK IDs
  # /////////////////////////////////////////////////////////////
  nranks = 1
  myrank = 0
  i_beg = 0
  nx = nx_glob
  left_rank = 0
  right_rank = 0
  # //////////////////////////////////////////////
  # END MPI DUMMY SECTION
  # //////////////////////////////////////////////

  # Vertical direction isn't MPI-ized,
  # so the rank's local values = the global values
  k_beg = 0
  nz = nz_glob
  mainproc = (myrank == 0)

  # Allocate the model data
  state = real3d("state", NUM_VARS, nz+2*hs, nx+2*hs)

  # Define the maximum stable time step based on an assumed maximum wind speed
  dt: real = min(dx,dz) / max_speed * cfl;

  # If I'm the main process in MPI, display some grid information
  if mainproc:
    print(f"nx_glob, nz_glob: {nx_glob} {nz_glob}\n")
    print(f"dx,dz: {dx} {dz}\n")
    print(f"dt: {dt}\n")

  # Want to make sure this info is displayed before further output
  # ierr = MPI_Barrier(MPI_COMM_WORLD);

  # Define quadrature weights and points
  nqpoints: int = 3
  qpoints = np.array([
      0.112701665379258311482073460022,
      0.500000000000000000000000000000,
      0.887298334620741688517926539980
    ], dtype=real)
  qweights = np.array([
      0.277777777777777777777777777779,
      0.444444444444444444444444444444,
      0.277777777777777777777777777779
    ], dtype=real)

  # //////////////////////////////////////////////////////////////////////////
  # Initialize the cell-averaged fluid state via Gauss-Legendre quadrature
  #//////////////////////////////////////////////////////////////////////////
  #///////////////////////////////////////////////
  # TODO: MAKE THESE 2 LOOPS A PARALLEL_FOR
  #///////////////////////////////////////////////
  for k in range(nz+2*hs):
    for i in range(nx+2*hs):
      # Initialize the state to zero
      for ll in range(NUM_VARS):
        state[ll,k,i] = 0.0
      
      # Use Gauss-Legendre quadrature to initialize a hydrostatic balance + temperature perturbation
      for kk in range(nqpoints):
        for ii in range(nqpoints):
          # Compute the x,z location within the global domain based on cell and quadrature index
          x: real = (i_beg + i-hs+0.5)*dx + (qpoints[ii]-0.5)*dx
          z: real = (k_beg + k-hs+0.5)*dz + (qpoints[kk]-0.5)*dz

          # Set the fluid state based on the user's specification
          if data_spec_int == DATA_SPEC_COLLISION:
            (r,u,w,t,hr,ht) = collision(x,z)
          if data_spec_int == DATA_SPEC_THERMAL:
            (r,u,w,t,hr,ht) = thermal(x,z)
          if data_spec_int == DATA_SPEC_GRAVITY_WAVES:
            (r,u,w,t,hr,ht) = gravity_waves(x,z)
          if data_spec_int == DATA_SPEC_DENSITY_CURRENT:
            (r,u,w,t,hr,ht) = density_current(x,z)
          if data_spec_int == DATA_SPEC_INJECTION:
            (r,u,w,t,hr,ht) = injection(x,z)

          # Store into the fluid state array
          state[ID_DENS,k,i] += r                         * qweights[ii]*qweights[kk]
          state[ID_UMOM,k,i] += (r+hr)*u                  * qweights[ii]*qweights[kk]
          state[ID_WMOM,k,i] += (r+hr)*w                  * qweights[ii]*qweights[kk]
          state[ID_RHOT,k,i] += ( (r+hr)*(t+ht) - hr*ht ) * qweights[ii]*qweights[kk]

  hy_dens_cell       = real1d("hy_dens_cell      ", nz+2*hs)
  hy_dens_theta_cell = real1d("hy_dens_theta_cell", nz+2*hs)
  hy_dens_int        = real1d("hy_dens_int       ", nz+1)
  hy_dens_theta_int  = real1d("hy_dens_theta_int ", nz+1)
  hy_pressure_int    = real1d("hy_pressure_int   ", nz+1)

  # Compute the hydrostatic background state over vertical cell averages
  # /////////////////////////////////////////////////
  # // TODO: MAKE THIS LOOP A PARALLEL_FOR
  # /////////////////////////////////////////////////
  for k in range(nz+2*hs):
    hy_dens_cell[k]       = 0.0
    hy_dens_theta_cell[k] = 0.0
    for kk in range(nqpoints):
      z = (k_beg + k-hs+0.5)*dz
      # real r, u, w, t, hr, ht; # formerly output arguments
      # Set the fluid state based on the user's specification
      if data_spec_int == DATA_SPEC_COLLISION:
        (r, u, w, t, hr, ht) = collision(0.0, z)
      if data_spec_int == DATA_SPEC_THERMAL:
        (r, u, w, t, hr, ht) = thermal(0.0, z)
      if data_spec_int == DATA_SPEC_GRAVITY_WAVES:
        (r, u, w, t, hr, ht) = gravity_waves(0.0, z)
      if data_spec_int == DATA_SPEC_DENSITY_CURRENT:
        (r, u, w, t, hr, ht) = density_current(0.0, z)
      if data_spec_int == DATA_SPEC_INJECTION:
        (r, u, w, t, hr, ht) = injection(0.0, z)
      
      hy_dens_cell[k]       = hy_dens_cell[k]       + hr    * qweights[kk];
      hy_dens_theta_cell[k] = hy_dens_theta_cell[k] + hr*ht * qweights[kk];

  # Compute the hydrostatic background state at vertical cell interfaces
  # /////////////////////////////////////////////////
  # // TODO: MAKE THIS LOOP A PARALLEL_FOR
  # /////////////////////////////////////////////////
  for k in range(nz+1):
    z = (k_beg + k)*dz
    # real r, u, w, t, hr, ht; # formerly output arguments
    if data_spec_int == DATA_SPEC_COLLISION:
      (r, u, w, t, hr, ht) = collision(0.0, z)
    if data_spec_int == DATA_SPEC_THERMAL:
      (r, u, w, t, hr, ht) = thermal(0.0, z)
    if data_spec_int == DATA_SPEC_GRAVITY_WAVES:
      (r, u, w, t, hr, ht) = gravity_waves(0.0, z)
    if data_spec_int == DATA_SPEC_DENSITY_CURRENT:
      (r, u, w, t, hr, ht) = density_current(0.0, z)
    if data_spec_int == DATA_SPEC_INJECTION:
      (r, u, w, t, hr, ht) = injection(0.0, z)
    
    hy_dens_int[k]       = hr
    hy_dens_theta_int[k] = hr*ht
    hy_pressure_int[k]   = C0*pow((hr*ht), gamm)

  hy_dens_cell       = realConst1d(hy_dens_cell      )
  hy_dens_theta_cell = realConst1d(hy_dens_theta_cell)
  hy_dens_int        = realConst1d(hy_dens_int       )
  hy_dens_theta_int  = realConst1d(hy_dens_theta_int )
  hy_pressure_int    = realConst1d(hy_pressure_int   )

  fixed_data = Fixed_data(nx, nz, i_beg, k_beg,
    nranks, myrank, left_rank, right_rank, mainproc,
    hy_dens_cell, hy_dens_theta_cell, hy_dens_int,
    hy_dens_theta_int, hy_pressure_int)

  return (state, fixed_data, dt)

# This test case is initially balanced but injects fast, cold air from the left boundary near the model top
# x and z are input coordinates at which to sample
# r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
# hr and ht are output background hydrostatic density and potential temperature at that location
def injection(x: real, z: real): # returns (r, u, w, t, hr, ht)
  (hr, ht) = hydro_const_theta(z)
  r = 0.0
  t = 0.0
  u = 0.0
  w = 0.0

  return (r, u, w, t, hr, ht)


# Initialize a density current (falling cold thermal that propagates along the model bottom)
# x and z are input coordinates at which to sample
# r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
# hr and ht are output background hydrostatic density and potential temperature at that location
def density_current(x: real, z: real): # returns (r, u, w, t, hr, ht)
  (hr, ht) = hydro_const_theta(z)
  r = 0.0
  t = 0.0
  u = 0.0
  w = 0.0
  # FIXME (mfh 2025/03/03) This was originally t = t + ... so perhaps t should also be an input parameter.
  # On the other hand, t was uninitialized before (which means this was probably incorrect in the original C++).
  t = sample_ellipse_cosine(x, z, -20.0, xlen/2, 5000.0, 4000.0, 2000.0)

  return (r, u, w, t, hr, ht)


# x and z are input coordinates at which to sample
# r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
# hr and ht are output background hydrostatic density and potential temperature at that location
def gravity_waves(x: real, z: real): # returns (r, u, w, t, hr, ht)
  (hr, ht) = hydro_const_bvfreq(z, 0.02)
  r = 0.0
  t = 0.0
  u = 15.0
  w = 0.0

  return (r, u, w, t, hr, ht)


# Rising thermal
# x and z are input coordinates at which to sample
# r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
# hr and ht are output background hydrostatic density and potential temperature at that location
def thermal(x: real, z: real): # returns (r, u, w, t, hr, ht)
  (hr, ht) = hydro_const_theta(z)
  r = 0.0
  t = 0.0
  u = 0.0
  w = 0.0
  # FIXME (mfh 2025/03/03) This was originally t = t + ... so perhaps t should also be an input parameter.
  # On the other hand, t was uninitialized before (which means this was probably incorrect in the original C++).
  t = sample_ellipse_cosine(x, z, 3.0, xlen/2, 2000.0, 2000.0, 2000.0)

  return (r, u, w, t, hr, ht)


# Colliding thermals
# x and z are input coordinates at which to sample
# r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
# hr and ht are output background hydrostatic density and potential temperature at that location
def collision(x: real, z: real): # returns (r, u, w, t, hr, ht)
  (hr, ht) = hydro_const_theta(z)
  r = 0.0
  t = 0.0
  u = 0.0
  w = 0.0
  # FIXME (mfh 2025/03/03) This was originally t = t + ... so perhaps t should also be an input parameter.
  # On the other hand, t was uninitialized before (which means this was probably incorrect in the original C++).
  t = sample_ellipse_cosine(x, z,  20.0, xlen/2, 2000.0, 2000.0, 2000.0) + \
      sample_ellipse_cosine(x, z, -20.0, xlen/2, 8000.0, 2000.0, 2000.0)

  return (r, u, w, t, hr, ht)


# Establish hydrostatic balance using constant potential temperature (thermally neutral atmosphere)
# z is the input coordinate
# Returns (r,t): r and t are the output background hydrostatic density and potential temperature
def hydro_const_theta(z: real): # returns (r, t)
  theta0: real = 300.0  # Background potential temperature
  exner0: real =   1.0  # Surface-level Exner pressure
  # Establish hydrostatic balance first using Exner pressure
  t = theta0                                  # Potential Temperature at z
  exner = exner0 - grav * z / (cp * theta0)   # Exner pressure at z
  p = p0 * pow(exner, (cp/rd))                # Pressure at z
  rt = pow((p / C0), (1. / gamm))             # rho*theta at z
  r = rt / t                                  # Density at z

  return (r, t)
}


# Establish hydrostatic balance using constant Brunt-Vaisala frequency
# z is the input coordinate
# bv_freq0 is the constant Brunt-Vaisala frequency (also an input)
# r and t are the output background hydrostatic density and potential temperature
def hydro_const_bvfreq(z: real, bv_freq0: real): # returns (r, t)
  theta0: real = 300.0  # Background potential temperature
  exner0: real =   1.0  # Surface-level Exner pressure
  t = theta0 * exp( bv_freq0*bv_freq0 / grav * z )                                     # Pot temp at z
  exner = exner0 - grav*grav / (cp * bv_freq0*bv_freq0) * (t - theta0) / (t * theta0)  # Exner pressure at z
  p = p0 * pow(exner,(cp/rd))                                                          # Pressure at z
  rt = pow((p / C0),(1. / gamm))                                                       # rho*theta at z
  r = rt / t                                                                           # Density at z

  return (r, t)


# Sample from an ellipse of a specified center, radius, and amplitude at a specified location
# x and z are input coordinates
# amp,x0,z0,xrad,zrad are input amplitude, center, and radius of the ellipse
def sample_ellipse_cosine(x: real, z: real, amp: real, x0: real, z0: real, xrad: real, zrad: real) -> real:
  # Compute distance from bubble center
  dist: real = sqrt( ((x-x0)/xrad)*((x-x0)/xrad) + ((z-z0)/zrad)*((z-z0)/zrad) ) * pi / 2.0
  # If the distance from bubble center is less than the radius, create a cos**2 profile
  if dist <= pi / 2.0:
    return amp * pow(cos(dist),2.)
  else:
    return 0.0


# Output the fluid state (state) to a NetCDF file at a given elapsed model time (etime)
# The file I/O uses parallel-netcdf, the only external library required for this mini-app.
# If it's too cumbersome, you can comment the I/O out, but you'll miss out on some potentially cool graphics
def output(
    state, # realConst3d
    etime, # real
    num_out, # int
    fixed_data # Fixed_data const&
  ) -> int: # num_out (updated)

  if mainproc:
    print("*** OUTPUT ***\n")

  # TODO (mfh 2025/03/04) Actually write to the output file.

  # Increment the number of outputs
  num_out = num_out + 1;

  return num_out


def finalize() -> None:
  pass


# Compute reduced quantities for error checking without resorting to the "ncdiff" tool
#void reductions( realConst3d state , double &mass , double &te , Fixed_data const &fixed_data ) {
def reductions(
    state # realConst3d, an input parameter
    fixed_data # Fixed_data const&, an input parameter
  ) -> tuple[double, double]: # mass, te

  nx                 = fixed_data.nx                
  nz                 = fixed_data.nz                
  hy_dens_cell       = fixed_data.hy_dens_cell      
  hy_dens_theta_cell = fixed_data.hy_dens_theta_cell

  mass: double = 0.0
  te: double   = 0.0
  for k in range(nz):
    for i in range(nx):
      r = state[ID_DENS,hs+k,hs+i] + hy_dens_cell[hs+k]                # Density
      u = state[ID_UMOM,hs+k,hs+i] / r                                 # U-wind
      w = state[ID_WMOM,hs+k,hs+i] / r                                 # W-wind
      th = ( state[ID_RHOT,hs+k,hs+i] + hy_dens_theta_cell[hs+k] ) / r # Potential Temperature (theta)
      p  = C0*pow(r*th,gamm)                               # Pressure
      t  = th / pow(p0/p,rd/cp)                            # Temperature
      ke = r*(u*u+w*w)                                     # Kinetic Energy
      ie = r*cv*t                                          # Internal Energy
      mass += r        *dx*dz # Accumulate domain mass
      te   += (ke + ie)*dx*dz # Accumulate domain total energy

  #double glob[2], loc[2];
  #loc[0] = mass;
  #loc[1] = te;
  #int ierr = MPI_Allreduce(loc,glob,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  #mass = glob[0];
  #te   = glob[1];

  return (mass, te)
