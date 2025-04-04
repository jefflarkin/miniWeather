#include "miniWeather_common.hpp"

std::unique_ptr<double[]>
make_unique_array_3d(host_memory_space, int X, int Y, int Z) {
  return std::make_unique<double[]>(X * Y * Z);
}

std::unique_ptr<double[]>
make_unique_array_1d(host_memory_space, int X) {
  return std::make_unique<double[]>(X);
}

test_case injection(double x , double z) {
  auto [hr, ht] = hydro_const_theta(z);
  double r = 0.0;
  double t = 0.0;
  double u = 0.0;
  double w = 0.0;
  return {r, u, w, t, hr, ht};
}

test_case density_current(double x , double z) {
  auto [hr, ht] = hydro_const_theta(z);
  double r = 0.0;
  double t = sample_ellipse_cosine(x, z, -20.0, xlen/2, 5000.0, 4000.0, 2000.0);
  double u = 0.0;
  double w = 0.0;
  return {r, u, w, t, hr, ht};
}

test_case gravity_waves(double x, double z) {
  auto [hr, ht] = hydro_const_bvfreq(z, 0.02);
  double r = 0.0;
  double t = 0.0;
  double u = 15.0;
  double w = 0.0;
  return {r, u, w, t, hr, ht};
}

test_case thermal(double x, double z) {
  auto [hr, ht] = hydro_const_theta(z);
  double r = 0.0;
  double t = sample_ellipse_cosine(x, z, 3.0, xlen/2,2000.0, 2000.0, 2000.0);
  double u = 0.0;
  double w = 0.0;
  return {r, u, w, t, hr, ht};
}

test_case collision(double x , double z) {
  auto [hr, ht] = hydro_const_theta(z);
  double r = 0.0;
  double t = 0.0;
  double u = 0.0;
  double w = 0.0;
  t = t + sample_ellipse_cosine(x, z,  20.0, xlen/2,2000.0, 2000.0, 2000.0);
  t = t + sample_ellipse_cosine(x, z, -20.0, xlen/2,8000.0, 2000.0, 2000.0);
  return {r, u, w, t, hr, ht};
}

test_case get_test_case(int data_spec, double x_, double z_) {
  if (data_spec == DATA_SPEC_COLLISION      ) { return collision(x_, z_); }
  if (data_spec == DATA_SPEC_THERMAL        ) { return thermal(x_, z_); }
  if (data_spec == DATA_SPEC_GRAVITY_WAVES  ) { return gravity_waves(x_, z_); }
  if (data_spec == DATA_SPEC_DENSITY_CURRENT) { return density_current(x_, z_); }
  if (data_spec == DATA_SPEC_INJECTION      ) { return injection(x_, z_); }
  assert(false);
  return test_case{};
}

r_t_pair hydro_const_theta(double z) {
  const double theta0 = 300.;  //Background potential temperature
  const double exner0 = 1.;    //Surface-level Exner pressure
  double       p,exner,rt;
  //Establish hydrostatic balance first using Exner pressure
  double t = theta0;                         //Potential Temperature at z
  exner = exner0 - grav * z / (cp * theta0); //Exner pressure at z
  p = p0 * pow(exner,(cp/rd));               //Pressure at z
  rt = pow((p / C0),(1. / gamm));            //rho*theta at z
  double r = rt / t;                         //Density at z

  return {r, t};
}

r_t_pair hydro_const_bvfreq(double z, double bv_freq0) {
  const double theta0 = 300.;  //Background potential temperature
  const double exner0 = 1.;    //Surface-level Exner pressure
  double       p, exner, rt;
  double t = theta0 * exp( bv_freq0*bv_freq0 / grav * z );                                    //Pot temp at z
  exner = exner0 - grav*grav / (cp * bv_freq0*bv_freq0) * (t - theta0) / (t * theta0); //Exner pressure at z
  p = p0 * pow(exner,(cp/rd));                                                         //Pressure at z
  rt = pow((p / C0), (1. / gamm));                                                  //rho*theta at z
  double r = rt / t;                                                                          //Density at z

  return {r, t};
}

double sample_ellipse_cosine( double x , double z , double amp , double x0 , double z0 , double xrad , double zrad ) {
  double dist;
  //Compute distance from bubble center
  dist = sqrt( ((x-x0)/xrad)*((x-x0)/xrad) + ((z-z0)/zrad)*((z-z0)/zrad) ) * pi / 2.0;
  //If the distance from bubble center is less than the radius, create a cos**2 profile
  if (dist <= pi / 2.0) {
    return amp * pow(cos(dist), 2.0);
  } else {
    return 0.;
  }
}

void finalize() {
  (void) MPI_Finalize();
}
