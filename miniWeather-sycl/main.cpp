
//////////////////////////////////////////////////////////////////////////////////////////
// miniWeather in SYCL
// Original author: Matt Norman <normanmr@ornl.gov>  , Oak Ridge National Laboratory
//
// This code simulates dry, stratified, compressible, non-hydrostatic fluid flows
//
//////////////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <sycl/sycl.hpp>

const double pi        = 3.14159265358979323846264338327;   //Pi
const double grav      = 9.8;                               //Gravitational acceleration (m / s^2)
const double cp        = 1004.;                             //Specific heat of dry air at constant pressure
const double cv        = 717.;                              //Specific heat of dry air at constant volume
const double rd        = 287.;                              //Dry air constant for equation of state (P=rho*rd*T)
const double p0        = 1.e5;                              //Standard pressure at the surface in Pascals
const double C0        = 27.5629410929725921310572974482;   //Constant to translate potential temperature into pressure (P=C0*(rho*theta)**gamma)
const double gamm      = 1.40027894002789400278940027894;   //gamma=cp/Rd , have to call this gamm because "gamma" is taken (I hate C so much)
//Define domain and stability-related constants
const double xlen      = 2.e4;    //Length of the domain in the x-direction (meters)
const double zlen      = 1.e4;    //Length of the domain in the z-direction (meters)
const double hv_beta   = 0.25;     //How strong to diffuse the solution: hv_beta \in [0:1]
const double cfl       = 1.50;    //"Courant, Friedrichs, Lewy" number (for numerical stability)
const double max_speed = 450;        //Assumed maximum wave speed during the simulation (speed of sound + speed of wind) (meter / sec)
const int hs        = 2;          //"Halo" size: number of cells beyond the MPI tasks's domain needed for a full "stencil" of information for reconstruction
const int sten_size = 4;          //Size of the stencil used for interpolation

//Parameters for indexing and flags
const int NUM_VARS = 4;           //Number of fluid state variables
const int ID_DENS  = 0;           //index for density ("rho")
const int ID_UMOM  = 1;           //index for momentum in the x-direction ("rho * u")
const int ID_WMOM  = 2;           //index for momentum in the z-direction ("rho * w")
const int ID_RHOT  = 3;           //index for density * potential temperature ("rho * theta")
const int DIR_X = 1;              //Integer constant to express that this operation is in the x-direction
const int DIR_Z = 2;              //Integer constant to express that this operation is in the z-direction
const int DATA_SPEC_COLLISION       = 1;
const int DATA_SPEC_THERMAL         = 2;
const int DATA_SPEC_MOUNTAIN        = 3;
const int DATA_SPEC_TURBULENCE      = 4;
const int DATA_SPEC_DENSITY_CURRENT = 5;
const int DATA_SPEC_INJECTION       = 6;

const int nqpoints = 3;
double qpoints [] = { 0.112701665379258311482073460022E0 , 0.500000000000000000000000000000E0 , 0.887298334620741688517926539980E0 };
double qweights[] = { 0.277777777777777777777777777779E0 , 0.444444444444444444444444444444E0 , 0.277777777777777777777777777779E0 };

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are initialized but remain static over the coure of the simulation
///////////////////////////////////////////////////////////////////////////////////////
double sim_time;              //total simulation time in seconds
double dt;                    //Model time step (seconds)
int    nx, nz;                //Number of local grid cells in the x- and z- dimensions for this MPI task
double dx, dz;                //Grid space length in x- and z-dimension (meters)
int    nx_glob, nz_glob;      //Number of total grid cells in the x- and z- dimensions
int    i_beg, k_beg;          //beginning index in the x- and z-directions for this MPI task
int    nranks, myrank;        //Number of MPI ranks and my rank id
int    left_rank, right_rank; //MPI Rank IDs that exist to my left and right in the global domain
int    masterproc;            //Am I the master process (rank == 0)?
double data_spec_int;         //Which data initialization to use
double *hy_dens_cell;         //hydrostatic density (vert cell avgs).   Dimensions: (1-hs:nz+hs)
double *hy_dens_theta_cell;   //hydrostatic rho*t (vert cell avgs).     Dimensions: (1-hs:nz+hs)
double *hy_dens_int;          //hydrostatic density (vert cell interf). Dimensions: (1:nz+1)
double *hy_dens_theta_int;    //hydrostatic rho*t (vert cell interf).   Dimensions: (1:nz+1)
double *hy_pressure_int;      //hydrostatic press (vert cell interf).   Dimensions: (1:nz+1)

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are dynamics over the course of the simulation
///////////////////////////////////////////////////////////////////////////////////////
double etime;                 //Elapsed model time
double output_counter;        //Helps determine when it's time to do output
//Runtime variable arrays
double *state;                //Fluid state.             Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
double *state_tmp;            //Fluid state.             Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
double *flux;                 //Cell interface fluxes.   Dimensions: (nx+1,nz+1,NUM_VARS)
double *tend;                 //Fluid state tendencies.  Dimensions: (nx,nz,NUM_VARS)
double *sendbuf_l;            //Buffer to send data to the left MPI rank
double *sendbuf_r;            //Buffer to send data to the right MPI rank
double *recvbuf_l;            //Buffer to receive data from the left MPI rank
double *recvbuf_r;            //Buffer to receive data from the right MPI rank
int    num_out = 0;           //The number of outputs performed so far
int    direction_switch = 1;
double mass0, te0;            //Initial domain totals for mass and total energy
double mass , te ;            //Domain totals for mass and total energy

//Establish hydrstatic balance using constant potential temperature (thermally neutral atmosphere)
//z is the input coordinate
//r and t are the output background hydrostatic density and potential temperature
void hydro_const_theta( double z , double &r , double &t ) {
  const double theta0 = 300.;  //Background potential temperature
  const double exner0 = 1.;    //Surface-level Exner pressure
  double       p,exner,rt;
  //Establish hydrostatic balance first using Exner pressure
  t = theta0;                                  //Potential Temperature at z
  exner = exner0 - grav * z / (cp * theta0);   //Exner pressure at z
  p = p0 * pow(exner,(cp/rd));                 //Pressure at z
  rt = pow((p / C0),(1. / gamm));             //rho*theta at z
  r = rt / t;                                  //Density at z
}


//Establish hydrstatic balance using constant Brunt-Vaisala frequency
//z is the input coordinate
//bv_freq0 is the constant Brunt-Vaisala frequency
//r and t are the output background hydrostatic density and potential temperature
void hydro_const_bvfreq( double z , double bv_freq0 , double &r , double &t ) {
  const double theta0 = 300.;  //Background potential temperature
  const double exner0 = 1.;    //Surface-level Exner pressure
  double       p, exner, rt;
  t = theta0 * exp( bv_freq0*bv_freq0 / grav * z );                                    //Pot temp at z
  exner = exner0 - grav*grav / (cp * bv_freq0*bv_freq0) * (t - theta0) / (t * theta0); //Exner pressure at z
  p = p0 * pow(exner,(cp/rd));                                                         //Pressure at z
  rt = pow((p / C0),(1. / gamm));                                                  //rho*theta at z
  r = rt / t;                                                                          //Density at z
}


//Sample from an ellipse of a specified center, radius, and amplitude at a specified location
//x and z are input coordinates
//amp,x0,z0,xrad,zrad are input amplitude, center, and radius of the ellipse
double sample_ellipse_cosine( double x , double z , double amp , double x0 , double z0 , double xrad , double zrad ) {
  double dist;
  //Compute distance from bubble center
  dist = sqrt( ((x-x0)/xrad)*((x-x0)/xrad) + ((z-z0)/zrad)*((z-z0)/zrad) ) * pi / 2.;
  //If the distance from bubble center is less than the radius, create a cos**2 profile
  if (dist <= pi / 2.) {
    return amp * pow(cos(dist),2.);
  } else {
    return 0.;
  }
}


//This test case is initially balanced but injects fast, cold air from the left boundary near the model top
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void injection( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
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
void density_current( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
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
void turbulence( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  hydro_const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
  // call random_number(u);
  // call random_number(w);
  // u = (u-0.5)*20;
  // w = (w-0.5)*20;
}


//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void mountain_waves( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
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
void thermal( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
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
void collision( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  hydro_const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
  t = t + sample_ellipse_cosine(x,z, 20.,xlen/2,2000.,2000.,2000.);
  t = t + sample_ellipse_cosine(x,z,-20.,xlen/2,8000.,2000.,2000.);
}


//Compute the time tendencies of the fluid state using forcing in the x-direction
//Since the halos are set in a separate routine, this will not require MPI
//First, compute the flux vector at each cell interface in the x-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
void compute_tendencies_x(
    const int hs,
    const int nx,
    const int nz,
    const double dx,
    double *d_state,
    double *d_flux,
    double *d_tend,
    double *d_hy_dens_cell,
    double *d_hy_dens_theta_cell,
    sycl::queue &q )
{
  sycl::range<3> flux_gws (1, (nz+15)/16*16, (nx+16)/16*16);
  sycl::range<3> flux_lws (1, 16, 16);

  //Compute the hyperviscosity coeficient
  double hv_coef = -hv_beta * dx / (16*dt);
  //Compute fluxes in the x-direction for each cell
  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class compute_flux_x>(
      sycl::nd_range<3>(flux_gws, flux_lws), [=] (sycl::nd_item<3> item) {
      int k = item.get_global_id(1);
      int i = item.get_global_id(2);
      double stencil[4], d3_vals[NUM_VARS], vals[NUM_VARS];

      if (i < nx+1 && k < nz) {
        //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
        for (int ll=0; ll<NUM_VARS; ll++) {
          for (int s=0; s < sten_size; s++) {
            int inds = ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+s;
            stencil[s] = d_state[inds];
          }
          //Fourth-order-accurate interpolation of the state
          vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
          //First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
          d3_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3];
        }

        //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
        double r = vals[ID_DENS] + d_hy_dens_cell[k+hs];
        double u = vals[ID_UMOM] / r;
        double w = vals[ID_WMOM] / r;
        double t = ( vals[ID_RHOT] + d_hy_dens_theta_cell[k+hs] ) / r;
        double p = C0*sycl::pow((r*t),gamm);

        //Compute the flux vector
        d_flux[ID_DENS*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u     - hv_coef*d3_vals[ID_DENS];
        d_flux[ID_UMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u*u+p - hv_coef*d3_vals[ID_UMOM];
        d_flux[ID_WMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u*w   - hv_coef*d3_vals[ID_WMOM];
        d_flux[ID_RHOT*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u*t   - hv_coef*d3_vals[ID_RHOT];
      }
    });
  });

  //Use the fluxes to compute tendencies for each cell
  sycl::range<3> tend_gws (NUM_VARS, (nz+15)/16*16, (nx+15)/16*16);
  sycl::range<3> tend_lws (1, 16, 16);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class compute_tend_x>(
      sycl::nd_range<3>(tend_gws, tend_lws), [=] (sycl::nd_item<3> item) {
      int ll = item.get_global_id(0);
      int k = item.get_global_id(1);
      int i = item.get_global_id(2);
      if (i < nx && k < nz) {
        int indt  = ll* nz   * nx    + k* nx    + i  ;
        int indf1 = ll*(nz+1)*(nx+1) + k*(nx+1) + i  ;
        int indf2 = ll*(nz+1)*(nx+1) + k*(nx+1) + i+1;
        d_tend[indt] = -( d_flux[indf2] - d_flux[indf1] ) / dx;
      }
    });
  });
}


//Compute the time tendencies of the fluid state using forcing in the z-direction
//Since the halos are set in a separate routine, this will not require MPI
//First, compute the flux vector at each cell interface in the z-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
void compute_tendencies_z(
    const int hs,
    const int nx,
    const int nz,
    const double dz,
    double *d_state,
    double *d_flux,
    double *d_tend,
    double *d_hy_dens_int,
    double *d_hy_dens_theta_int,
    double *d_hy_pressure_int,
    sycl::queue &q )
{
  //Compute the hyperviscosity coeficient
  double hv_coef = -hv_beta * dz / (16*dt);

  //Compute fluxes in the z-direction for each cell

  sycl::range<3> flux_gws (1, (nz+16)/16*16, (nx+15)/16*16);
  sycl::range<3> flux_lws (1, 16, 16);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class compute_flux_z>(
      sycl::nd_range<3>(flux_gws, flux_lws), [=] (sycl::nd_item<3> item) {
      int k = item.get_global_id(1);
      int i = item.get_global_id(2);
      double stencil[4], d3_vals[NUM_VARS], vals[NUM_VARS];

      if (i < nx && k < nz+1) {
        //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
        for (int ll=0; ll<NUM_VARS; ll++) {
          for (int s=0; s<sten_size; s++) {
            int inds = ll*(nz+2*hs)*(nx+2*hs) + (k+s)*(nx+2*hs) + i+hs;
            stencil[s] = d_state[inds];
          }
          //Fourth-order-accurate interpolation of the state
          vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
          //First-order-accurate interpolation of the third spatial derivative of the state
          d3_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3];
        }

        //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
        double r = vals[ID_DENS] + d_hy_dens_int[k];
        double u = vals[ID_UMOM] / r;
        double w = vals[ID_WMOM] / r;
        double t = ( vals[ID_RHOT] + d_hy_dens_theta_int[k] ) / r;
        double p = C0*sycl::pow((r*t),gamm) - d_hy_pressure_int[k];
        //Enforce vertical boundary condition and exact mass conservation
        if (k == 0 || k == nz) {
          w                = 0;
          d3_vals[ID_DENS] = 0;
        }

        //Compute the flux vector with hyperviscosity
        d_flux[ID_DENS*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w     - hv_coef*d3_vals[ID_DENS];
        d_flux[ID_UMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w*u   - hv_coef*d3_vals[ID_UMOM];
        d_flux[ID_WMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w*w+p - hv_coef*d3_vals[ID_WMOM];
        d_flux[ID_RHOT*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w*t   - hv_coef*d3_vals[ID_RHOT];
      }
    });
  });

  //Use the fluxes to compute tendencies for each cell
  sycl::range<3> tend_gws (NUM_VARS, (nz+15)/16*16, (nx+15)/16*16);
  sycl::range<3> tend_lws (1, 16, 16);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class compute_tend_z>(
      sycl::nd_range<3>(tend_gws, tend_lws), [=] (sycl::nd_item<3> item) {
      int ll = item.get_global_id(0);
      int k = item.get_global_id(1);
      int i = item.get_global_id(2);
      if (i < nx && k < nz) {
        int indt  = ll* nz   * nx    + k* nx    + i  ;
        int indf1 = ll*(nz+1)*(nx+1) + (k  )*(nx+1) + i;
        int indf2 = ll*(nz+1)*(nx+1) + (k+1)*(nx+1) + i;
        d_tend[indt] = -( d_flux[indf2] - d_flux[indf1] ) / dz;
        if (ll == ID_WMOM) {
          int inds = ID_DENS*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
          d_tend[indt] = d_tend[indt] - d_state[inds]*grav;
        }
      }
    });
  });
}


//Set this MPI task's halo values in the x-direction. This routine will require MPI
void set_halo_values_x(
    const int hs,
    const int nx,
    const int nz,
    const int k_beg,
    const double dz,
    double *d_state,
    double *d_hy_dens_cell,
    double *d_hy_dens_theta_cell,
    double *d_sendbuf_l,
    double *d_sendbuf_r,
    double *d_recvbuf_l,
    double *d_recvbuf_r,
    sycl::queue &q )
{
  int ierr;
  MPI_Request req_r[2], req_s[2];

  //Prepost receives
  ierr = MPI_Irecv(recvbuf_l,hs*nz*NUM_VARS,MPI_DOUBLE, left_rank,0,MPI_COMM_WORLD,&req_r[0]);
  ierr = MPI_Irecv(recvbuf_r,hs*nz*NUM_VARS,MPI_DOUBLE,right_rank,1,MPI_COMM_WORLD,&req_r[1]);

  //Pack the send buffers
  sycl::range<3> buffer_gws (NUM_VARS, (nz+15)/16*16, (hs+15)/16*16);
  sycl::range<3> buffer_lws (1, 16, 16);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class pack_send_buf>(
      sycl::nd_range<3>(buffer_gws, buffer_lws), [=] (sycl::nd_item<3> item) {
      int ll = item.get_global_id(0);
      int k = item.get_global_id(1);
      int s = item.get_global_id(2);
      if (s < hs && k < nz) {
        d_sendbuf_l[ll*nz*hs + k*hs + s] = d_state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + hs+s];
        d_sendbuf_r[ll*nz*hs + k*hs + s] = d_state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + nx+s];
      }
    });
  });

  q.memcpy(sendbuf_l, d_sendbuf_l, sizeof(double)*hs*nz*NUM_VARS);
  q.memcpy(sendbuf_r, d_sendbuf_r, sizeof(double)*hs*nz*NUM_VARS);
  q.wait();

  //#pragma omp target update from(sendbuf_l[:nz*hs*NUM_VARS],sendbuf_r[:nz*hs*NUM_VARS])

  //Fire off the sends
  ierr = MPI_Isend(sendbuf_l,hs*nz*NUM_VARS,MPI_DOUBLE, left_rank,1,MPI_COMM_WORLD,&req_s[0]);
  ierr = MPI_Isend(sendbuf_r,hs*nz*NUM_VARS,MPI_DOUBLE,right_rank,0,MPI_COMM_WORLD,&req_s[1]);

  //Wait for receives to finish
  ierr = MPI_Waitall(2,req_r,MPI_STATUSES_IGNORE);

  q.memcpy(d_recvbuf_l, recvbuf_l, sizeof(double)*hs*nz*NUM_VARS);
  q.memcpy(d_recvbuf_r, recvbuf_r, sizeof(double)*hs*nz*NUM_VARS);

  //Unpack the receive buffers
  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class unpack_recv_buf>(
      sycl::nd_range<3>(buffer_gws, buffer_lws), [=] (sycl::nd_item<3> item) {
      int ll = item.get_global_id(0);
      int k = item.get_global_id(1);
      int s = item.get_global_id(2);
      if (s < hs && k < nz) {
        d_state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + s      ] = d_recvbuf_l[ll*nz*hs + k*hs + s];
        d_state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + nx+hs+s] = d_recvbuf_r[ll*nz*hs + k*hs + s];
      }
    });
  });

  //Wait for sends to finish
  ierr = MPI_Waitall(2,req_s,MPI_STATUSES_IGNORE);

  if (data_spec_int == DATA_SPEC_INJECTION) {
    if (myrank == 0) {
      sycl::range<3> inj_gws (1, (nz+15)/16*16, (hs+15)/16*16);
      sycl::range<3> inj_lws (1, 16, 16);

      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class update_state_x>(
          sycl::nd_range<3>(inj_gws, inj_lws), [=] (sycl::nd_item<3> item) {
          int k = item.get_global_id(1);
          int i = item.get_global_id(2);
          if (i < hs && k < nz) {
            double z = (k_beg + k+0.5)*dz;
            if (sycl::fabs(z-3*zlen/4) <= zlen/16) {
              int ind_r = ID_DENS*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i;
              int ind_u = ID_UMOM*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i;
              int ind_t = ID_RHOT*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i;
              d_state[ind_u] = (d_state[ind_r]+d_hy_dens_cell[k+hs]) * 50.;
              d_state[ind_t] = (d_state[ind_r]+d_hy_dens_cell[k+hs]) * 298. - d_hy_dens_theta_cell[k+hs];
            }
          }
        });
      });
    }
  }
}


//Set this MPI task's halo values in the z-direction. This does not require MPI because there is no MPI
//decomposition in the vertical direction
void set_halo_values_z(
    const int hs, const int nx, const int nz,
    const int i_beg,
    const double dx,
    const int data_spec_int,
    double *d_state,
    sycl::queue &q )
{
  const double mnt_width = xlen/8;

  sycl::range<3> gws (1, (NUM_VARS+15)/16*16, (nx+2*hs+15)/16*16);
  sycl::range<3> lws (1, 16, 16);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class update_state_z>(
      sycl::nd_range<3>(gws, lws), [=] (sycl::nd_item<3> item) {
      int ll = item.get_global_id(1);
      int i = item.get_global_id(2);
      if (i < nx+2*hs && ll < NUM_VARS) {
        if (ll == ID_WMOM) {
          d_state[ll*(nz+2*hs)*(nx+2*hs) + (0      )*(nx+2*hs) + i] = 0.;
          d_state[ll*(nz+2*hs)*(nx+2*hs) + (1      )*(nx+2*hs) + i] = 0.;
          d_state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs  )*(nx+2*hs) + i] = 0.;
          d_state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs+1)*(nx+2*hs) + i] = 0.;
          //Impose the vertical momentum effects of an artificial cos^2 mountain at the lower boundary
          if (data_spec_int == DATA_SPEC_MOUNTAIN) {
            double x = (i_beg+i-hs+0.5)*dx;
            if ( sycl::fabs(x-xlen/4) < mnt_width ) {
              double xloc = (x-(xlen/4)) / mnt_width;
              //Compute the derivative of the fake mountain
              double mnt_deriv = -pi*sycl::cos(pi*xloc/2)*sycl::sin(pi*xloc/2)*10/dx;
              //w = (dz/dx)*u
              d_state[ID_WMOM*(nz+2*hs)*(nx+2*hs) + (0)*(nx+2*hs) + i] = mnt_deriv*d_state[ID_UMOM*(nz+2*hs)*(nx+2*hs) + hs*(nx+2*hs) + i];
              d_state[ID_WMOM*(nz+2*hs)*(nx+2*hs) + (1)*(nx+2*hs) + i] = mnt_deriv*d_state[ID_UMOM*(nz+2*hs)*(nx+2*hs) + hs*(nx+2*hs) + i];
            }
          }
        } else {
          d_state[ll*(nz+2*hs)*(nx+2*hs) + (0      )*(nx+2*hs) + i] = d_state[ll*(nz+2*hs)*(nx+2*hs) + (hs     )*(nx+2*hs) + i];
          d_state[ll*(nz+2*hs)*(nx+2*hs) + (1      )*(nx+2*hs) + i] = d_state[ll*(nz+2*hs)*(nx+2*hs) + (hs     )*(nx+2*hs) + i];
          d_state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs  )*(nx+2*hs) + i] = d_state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs-1)*(nx+2*hs) + i];
          d_state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs+1)*(nx+2*hs) + i] = d_state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs-1)*(nx+2*hs) + i];
        }
      }
    });
  });
}


void init( int *argc , char ***argv ) {
  int    i, k, ii, kk, ll, ierr, inds, i_end;
  double x, z, r, u, w, t, hr, ht, nper;

  ierr = MPI_Init(argc,argv);

  //Set the cell grid size
  dx = xlen / nx_glob;
  dz = zlen / nz_glob;

  ierr = MPI_Comm_size(MPI_COMM_WORLD,&nranks);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  nper = ( (double) nx_glob ) / nranks;
  i_beg = round( nper* (myrank)    );
  i_end = round( nper*((myrank)+1) )-1;
  nx = i_end - i_beg + 1;
  left_rank  = myrank - 1;
  if (left_rank == -1) left_rank = nranks-1;
  right_rank = myrank + 1;
  if (right_rank == nranks) right_rank = 0;


  ////////////////////////////////////////////////////////////////////////////////
  // YOU DON'T NEED TO ALTER ANYTHING BELOW THIS POINT IN THE CODE
  ////////////////////////////////////////////////////////////////////////////////

  //Vertical direction isn't MPI-ized, so the rank's local values = the global values
  k_beg = 0;
  nz = nz_glob;
  masterproc = (myrank == 0);

  //Allocate the model data
  state              = (double *) malloc( (nx+2*hs)*(nz+2*hs)*NUM_VARS*sizeof(double) );
  state_tmp          = (double *) malloc( (nx+2*hs)*(nz+2*hs)*NUM_VARS*sizeof(double) );
  flux               = (double *) malloc( (nx+1)*(nz+1)*NUM_VARS*sizeof(double) );
  tend               = (double *) malloc( nx*nz*NUM_VARS*sizeof(double) );
  hy_dens_cell       = (double *) malloc( (nz+2*hs)*sizeof(double) );
  hy_dens_theta_cell = (double *) malloc( (nz+2*hs)*sizeof(double) );
  hy_dens_int        = (double *) malloc( (nz+1)*sizeof(double) );
  hy_dens_theta_int  = (double *) malloc( (nz+1)*sizeof(double) );
  hy_pressure_int    = (double *) malloc( (nz+1)*sizeof(double) );
  sendbuf_l          = (double *) malloc( hs*nz*NUM_VARS*sizeof(double) );
  sendbuf_r          = (double *) malloc( hs*nz*NUM_VARS*sizeof(double) );
  recvbuf_l          = (double *) malloc( hs*nz*NUM_VARS*sizeof(double) );
  recvbuf_r          = (double *) malloc( hs*nz*NUM_VARS*sizeof(double) );

  //Define the maximum stable time step based on an assumed maximum wind speed
  dt = fmin(dx,dz) / max_speed * cfl;
  //Set initial elapsed model time and output_counter to zero
  etime = 0.;
  output_counter = 0.;

  //If I'm the master process in MPI, display some grid information
  if (masterproc) {
    printf( "nx_glob, nz_glob: %d %d\n", nx_glob, nz_glob);
    printf( "dx,dz: %lf %lf\n",dx,dz);
    printf( "dt: %lf\n",dt);
  }
  //Want to make sure this info is displayed before further output
  ierr = MPI_Barrier(MPI_COMM_WORLD);

  //////////////////////////////////////////////////////////////////////////
  // Initialize the cell-averaged fluid state via Gauss-Legendre quadrature
  //////////////////////////////////////////////////////////////////////////
  for (k=0; k<nz+2*hs; k++) {
    for (i=0; i<nx+2*hs; i++) {
      //Initialize the state to zero
      for (ll=0; ll<NUM_VARS; ll++) {
        inds = ll*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
        state[inds] = 0.;
      }
      //Use Gauss-Legendre quadrature to initialize a hydrostatic balance + temperature perturbation
      for (kk=0; kk<nqpoints; kk++) {
        for (ii=0; ii<nqpoints; ii++) {
          //Compute the x,z location within the global domain based on cell and quadrature index
          x = (i_beg + i-hs+0.5)*dx + (qpoints[ii]-0.5)*dx;
          z = (k_beg + k-hs+0.5)*dz + (qpoints[kk]-0.5)*dz;

          //Set the fluid state based on the user's specification
          if (data_spec_int == DATA_SPEC_COLLISION      ) { collision      (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_THERMAL        ) { thermal        (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_MOUNTAIN       ) { mountain_waves (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_TURBULENCE     ) { turbulence     (x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) { density_current(x,z,r,u,w,t,hr,ht); }
          if (data_spec_int == DATA_SPEC_INJECTION      ) { injection      (x,z,r,u,w,t,hr,ht); }

          //Store into the fluid state array
          inds = ID_DENS*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
          state[inds] = state[inds] + r                         * qweights[ii]*qweights[kk];
          inds = ID_UMOM*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
          state[inds] = state[inds] + (r+hr)*u                  * qweights[ii]*qweights[kk];
          inds = ID_WMOM*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
          state[inds] = state[inds] + (r+hr)*w                  * qweights[ii]*qweights[kk];
          inds = ID_RHOT*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
          state[inds] = state[inds] + ( (r+hr)*(t+ht) - hr*ht ) * qweights[ii]*qweights[kk];
        }
      }
      for (ll=0; ll<NUM_VARS; ll++) {
        inds = ll*(nz+2*hs)*(nx+2*hs) + k*(nx+2*hs) + i;
        state_tmp[inds] = state[inds];
      }
    }
  }
  //Compute the hydrostatic background state over vertical cell averages
  for (k=0; k<nz+2*hs; k++) {
    hy_dens_cell      [k] = 0.;
    hy_dens_theta_cell[k] = 0.;
    for (kk=0; kk<nqpoints; kk++) {
      z = (k_beg + k-hs+0.5)*dz;
      //Set the fluid state based on the user's specification
      if (data_spec_int == DATA_SPEC_COLLISION      ) { collision      (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_THERMAL        ) { thermal        (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_MOUNTAIN       ) { mountain_waves (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_TURBULENCE     ) { turbulence     (0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) { density_current(0.,z,r,u,w,t,hr,ht); }
      if (data_spec_int == DATA_SPEC_INJECTION      ) { injection      (0.,z,r,u,w,t,hr,ht); }
      hy_dens_cell      [k] = hy_dens_cell      [k] + hr    * qweights[kk];
      hy_dens_theta_cell[k] = hy_dens_theta_cell[k] + hr*ht * qweights[kk];
    }
  }
  //Compute the hydrostatic background state at vertical cell interfaces
  for (k=0; k<nz+1; k++) {
    z = (k_beg + k)*dz;
    if (data_spec_int == DATA_SPEC_COLLISION      ) { collision      (0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_THERMAL        ) { thermal        (0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_MOUNTAIN       ) { mountain_waves (0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_TURBULENCE     ) { turbulence     (0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_DENSITY_CURRENT) { density_current(0.,z,r,u,w,t,hr,ht); }
    if (data_spec_int == DATA_SPEC_INJECTION      ) { injection      (0.,z,r,u,w,t,hr,ht); }
    hy_dens_int      [k] = hr;
    hy_dens_theta_int[k] = hr*ht;
    hy_pressure_int  [k] = C0*pow((hr*ht),gamm);
  }
}



void finalize() {
  int ierr;
  free( state );
  free( state_tmp );
  free( flux );
  free( tend );
  free( hy_dens_cell );
  free( hy_dens_theta_cell );
  free( hy_dens_int );
  free( hy_dens_theta_int );
  free( hy_pressure_int );
  free( sendbuf_l );
  free( sendbuf_r );
  free( recvbuf_l );
  free( recvbuf_r );
  ierr = MPI_Finalize();
}


//Compute reduced quantities for error checking without resorting to the "ncdiff" tool
//#pragma omp target teams distribute parallel for collapse(2) reduction(+:mass,te)

static inline void atomicAdd(double& val, const double delta)
{
  sycl::atomic_ref<double, sycl::memory_order::relaxed,
                   sycl::memory_scope::device,
                   sycl::access::address_space::global_space> ref(val);
  ref.fetch_add(delta);
}

void reductions(
    double &mass,
    double &te,
    const int hs,
    const int nx,
    const int nz,
    const double dx,
    const double dz,
    const double *d_state,
    const double *d_hy_dens_cell,
    const double *d_hy_dens_theta_cell,
    sycl::queue &q )
{
  double* d_mass, *d_te;
  d_mass = sycl::malloc_device<double>(1, q);
  d_te = sycl::malloc_device<double>(1, q);

  sycl::range<3> gws (1, (nz+15)/16*16, (nx+15)/16*16);
  sycl::range<3> lws (1, 16, 16);

  q.memset(d_mass, 0, sizeof(double));
  q.memset(d_te, 0, sizeof(double));

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class reduce>(
      sycl::nd_range<3>(gws, lws), [=] (sycl::nd_item<3> item) {
      int k = item.get_global_id(1);
      int i = item.get_global_id(2);
      if (k < nz && i < nx) {
        int ind_r = ID_DENS*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
        int ind_u = ID_UMOM*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
        int ind_w = ID_WMOM*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
        int ind_t = ID_RHOT*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
        double r  =  d_state[ind_r] + d_hy_dens_cell[hs+k];           // Density
        double u  =  d_state[ind_u] / r;                              // U-wind
        double w  =  d_state[ind_w] / r;                              // W-wind
        double th = ( d_state[ind_t] + d_hy_dens_theta_cell[hs+k] ) / r; // Potential Temperature (theta)
        double p  = C0*sycl::pow(r*th,gamm);                               // Pressure
        double t  = th / sycl::pow(p0/p,rd/cp);                            // Temperature
        double ke = r*(u*u+w*w);                                     // Kinetic Energy
        double ie = r*cv*t;                                          // Internal Energy

        // mass += r        *dx*dz; // Accumulate domain mass
        // te   += (ke + ie)*dx*dz; // Accumulate domain total energy
        atomicAdd(d_mass[0], r*dx*dz);
        atomicAdd(d_te[0], (ke+ie)*dx*dz);
      }
    });
  });

  double glob[2], loc[2];

  q.memcpy(loc, d_mass, sizeof(double));
  q.memcpy(loc+1, d_te, sizeof(double));
  q.wait();

  int ierr = MPI_Allreduce(loc,glob,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

  free(d_mass, q);
  free(d_te, q);

  mass = glob[0];
  te   = glob[1];
}

//Perform a single semi-discretized step in time with the form:
//state_out = state_init + dt * rhs(state_forcing)
//Meaning the step starts from state_init, computes the rhs using state_forcing, and stores the result in state_out
void semi_discrete_step(
    const int hs,
    const int nx,
    const int nz,
    const int k_beg,
    const int i_beg,
    const double dx ,
    const double dz ,
    const double dt ,
    int dir ,
    const int data_spec_int,
    double *d_state_init ,
    double *d_state_forcing ,
    double *d_state_out,
    double *d_flux ,
    double *d_tend,
    double *d_hy_dens_cell ,
    double *d_hy_dens_theta_cell ,
    double *d_hy_dens_int ,
    double *d_hy_dens_theta_int ,
    double *d_hy_pressure_int ,
    double *d_sendbuf_l ,
    double *d_sendbuf_r ,
    double *d_recvbuf_l ,
    double *d_recvbuf_r ,
    sycl::queue &q  )
{
  if (dir == DIR_X) {
    //Set the halo values for this MPI task's fluid state in the x-direction
    set_halo_values_x(
        hs,
        nx,
        nz,
        k_beg,
        dz,
        d_state_forcing,
        d_hy_dens_cell ,
        d_hy_dens_theta_cell ,
        d_sendbuf_l,
        d_sendbuf_r,
        d_recvbuf_l,
        d_recvbuf_r,
        q);
    //Compute the time tendencies for the fluid state in the x-direction
    compute_tendencies_x(hs, nx, nz, dx, d_state_forcing, d_flux, d_tend,
                         d_hy_dens_cell, d_hy_dens_theta_cell, q);
  } else if (dir == DIR_Z) {
    //Set the halo values for this MPI task's fluid state in the z-direction
    set_halo_values_z(hs, nx, nz, i_beg, dx, data_spec_int, d_state_forcing, q);
    //Compute the time tendencies for the fluid state in the z-direction
    compute_tendencies_z(hs, nx, nz, dz, d_state_forcing, d_flux, d_tend,
                         d_hy_dens_int,d_hy_dens_theta_int,d_hy_pressure_int,q);
  }

  //Apply the tendencies to the fluid state
  sycl::range<3> tend_gws (NUM_VARS, (nz+15)/16*16, (nx+15)/16*16);
  sycl::range<3> tend_lws (1, 16, 16);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class update_fluid_state>(
      sycl::nd_range<3>(tend_gws, tend_lws), [=] (sycl::nd_item<3> item) {
      int ll = item.get_global_id(0);
      int k = item.get_global_id(1);
      int i = item.get_global_id(2);
      if (i < nx && k < nz) {
        int inds = ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
        int indt = ll*nz*nx + k*nx + i;
        d_state_out[inds] = d_state_init[inds] + dt * d_tend[indt];
      }
    });
  });
}


//Performs a single dimensionally split time step using a simple low-storate three-stage Runge-Kutta time integrator
//The dimensional splitting is a second-order-accurate alternating Strang splitting in which the
//order of directions is alternated each time step.
//The Runge-Kutta method used here is defined as follows:
// q*     = q[n] + dt/3 * rhs(q[n])
// q**    = q[n] + dt/2 * rhs(q*  )
// q[n+1] = q[n] + dt/1 * rhs(q** )
void perform_timestep(
    double *d_state ,
    double *d_state_tmp ,
    double *d_flux ,
    double *d_tend ,
    double *d_hy_dens_cell ,
    double *d_hy_dens_theta_cell ,
    double *d_hy_dens_int ,
    double *d_hy_dens_theta_int ,
    double *d_hy_pressure_int ,
    double *d_sendbuf_l ,
    double *d_sendbuf_r ,
    double *d_recvbuf_l ,
    double *d_recvbuf_r ,
    const double dt,
    sycl::queue &q ) {

// semi discrete step
#define SEMI_DSTEP(dt, dir, state, next_state) \
    semi_discrete_step(hs, nx, nz, k_beg, i_beg, dx, dz, dt, dir ,\
		    data_spec_int, d_state, state, \
		    next_state, d_flux, d_tend, \
		    d_hy_dens_cell, d_hy_dens_theta_cell,\
		    d_hy_dens_int, d_hy_dens_theta_int,\
		    d_hy_pressure_int, d_sendbuf_l,\
		    d_sendbuf_r, d_recvbuf_l, d_recvbuf_r, q);

  if (direction_switch) {
    SEMI_DSTEP(dt/3, DIR_X, d_state, d_state_tmp)
    SEMI_DSTEP(dt/2, DIR_X, d_state_tmp, d_state_tmp)
    SEMI_DSTEP(dt/1, DIR_X, d_state_tmp, d_state)
    SEMI_DSTEP(dt/3, DIR_Z, d_state, d_state_tmp)
    SEMI_DSTEP(dt/2, DIR_Z, d_state_tmp, d_state_tmp)
    SEMI_DSTEP(dt/1, DIR_Z, d_state_tmp, d_state)
  } else {
    SEMI_DSTEP(dt/3, DIR_Z, d_state, d_state_tmp)
    SEMI_DSTEP(dt/2, DIR_Z, d_state_tmp, d_state_tmp)
    SEMI_DSTEP(dt/1, DIR_Z, d_state_tmp, d_state)
    SEMI_DSTEP(dt/3, DIR_X, d_state, d_state_tmp)
    SEMI_DSTEP(dt/2, DIR_X, d_state_tmp, d_state_tmp)
    SEMI_DSTEP(dt/1, DIR_X, d_state_tmp, d_state)
  }

  if (direction_switch) { direction_switch = 0; } else { direction_switch = 1; }
}

///////////////////////////////////////////////////////////////////////////////////////
// THE MAIN PROGRAM STARTS HERE
///////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  ///////////////////////////////////////////////////////////////////////////////////////
  // BEGIN USER-CONFIGURABLE PARAMETERS
  ///////////////////////////////////////////////////////////////////////////////////////
  //The x-direction length is twice as long as the z-direction length
  //So, you'll want to have nx_glob be twice as large as nz_glob
  nx_glob = NX;               //Number of total cells in the x-dirction
  nz_glob = NZ;               //Number of total cells in the z-dirction
  sim_time = SIM_TIME;        //How many seconds to run the simulation
  data_spec_int = DATA_SPEC;  //How to initialize the data
  ///////////////////////////////////////////////////////////////////////////////////////
  // END USER-CONFIGURABLE PARAMETERS
  ///////////////////////////////////////////////////////////////////////////////////////

  init( &argc , &argv );

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  const int state_size = (nz+2*hs)*(nx+2*hs)*NUM_VARS;
  const int state_size_byte = (nz+2*hs)*(nx+2*hs)*NUM_VARS*sizeof(double);
  double *d_state_tmp = sycl::malloc_device<double>(state_size, q);
  q.memcpy(d_state_tmp, state, state_size_byte);

  double *d_state = sycl::malloc_device<double>(state_size, q);
  q.memcpy(d_state, state, state_size_byte);

  double *d_hy_dens_cell = sycl::malloc_device<double>(nz+2*hs, q);
  q.memcpy(d_hy_dens_cell, hy_dens_cell, (nz+2*hs)*sizeof(double));

  double *d_hy_dens_theta_cell = sycl::malloc_device<double>(nz+2*hs, q);
  q.memcpy(d_hy_dens_theta_cell, hy_dens_theta_cell, (nz+2*hs)*sizeof(double));

  double *d_hy_dens_int = sycl::malloc_device<double>(nz+1, q);
  q.memcpy(d_hy_dens_int, hy_dens_int, (nz+1)*sizeof(double));

  double *d_hy_dens_theta_int = sycl::malloc_device<double>(nz+1, q);
  q.memcpy(d_hy_dens_theta_int, hy_dens_theta_int, (nz+1)*sizeof(double));

  double *d_hy_pressure_int = sycl::malloc_device<double>(nz+1, q);
  q.memcpy(d_hy_pressure_int, hy_pressure_int, (nz+1)*sizeof(double));

  double *d_flux = sycl::malloc_device<double>((nz+1)*(nx+1)*NUM_VARS, q);

  double *d_tend = sycl::malloc_device<double>(nz*nx*NUM_VARS, q);

  double *d_sendbuf_l = sycl::malloc_device<double>(hs*nz*NUM_VARS, q);
  double *d_sendbuf_r = sycl::malloc_device<double>(hs*nz*NUM_VARS, q);
  double *d_recvbuf_l = sycl::malloc_device<double>(hs*nz*NUM_VARS, q);
  double *d_recvbuf_r = sycl::malloc_device<double>(hs*nz*NUM_VARS, q);

  //Initial reductions for mass, kinetic energy, and total energy
  reductions(mass0, te0, hs, nx, nz, dx, dz, d_state, d_hy_dens_cell, d_hy_dens_theta_cell, q);

  ////////////////////////////////////////////////////
  // MAIN TIME STEP LOOP
  ////////////////////////////////////////////////////
  auto c_start = clock();

  while (etime < sim_time) {
    //If the time step leads to exceeding the simulation time, shorten it for the last step
    if (etime + dt > sim_time) { dt = sim_time - etime; }
    //Perform a single time step
    perform_timestep(
        d_state,
        d_state_tmp,
        d_flux,
        d_tend,
        d_hy_dens_cell,
        d_hy_dens_theta_cell,
        d_hy_dens_int,
        d_hy_dens_theta_int,
        d_hy_pressure_int,
        d_sendbuf_l,
        d_sendbuf_r,
        d_recvbuf_l,
        d_recvbuf_r,
        dt,
        q);

    //Update the elapsed time and output counter
    etime = etime + dt;
  }

  auto c_end = clock();
  if (masterproc)
    printf("Total main time step loop: %lf sec\n", ( (double) (c_end-c_start) ) / CLOCKS_PER_SEC);

  //Final reductions for mass, kinetic energy, and total energy
  reductions(mass, te, hs, nx, nz, dx, dz, d_state, d_hy_dens_cell, d_hy_dens_theta_cell, q);

  printf( "d_mass: %le\n" , (mass - mass0) / mass0 );
  printf( "d_te:   %le\n" , (te   - te0  ) / te0   );

  finalize();

  sycl::free(d_state, q);
  sycl::free(d_state_tmp, q);
  sycl::free(d_flux, q);
  sycl::free(d_tend, q);
  sycl::free(d_hy_dens_cell, q);
  sycl::free(d_hy_dens_theta_cell, q);
  sycl::free(d_hy_dens_int, q);
  sycl::free(d_hy_dens_theta_int, q);
  sycl::free(d_hy_pressure_int, q);
  sycl::free(d_sendbuf_l, q);
  sycl::free(d_sendbuf_r, q);
  sycl::free(d_recvbuf_l, q);
  sycl::free(d_recvbuf_r, q);

  return 0;
}
