#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "common.h"

//define the data set size (cubic volume)
#define DATAXSIZE 600
#define DATAYSIZE 600
#define DATAZSIZE 600

typedef double nRarray[DATAYSIZE][DATAXSIZE];

// square
#define SQ(x) ((x)*(x))

#ifdef VERIFY
#include <string.h>
#include "reference.h"
#endif

double dFphi(double phi, double u, double lambda)
{
  return (-phi*(1.0-phi*phi)+lambda*u*(1.0-phi*phi)*(1.0-phi*phi));
}


double GradientX(double phi[][DATAYSIZE][DATAXSIZE], 
                 double dx, double dy, double dz, int x, int y, int z)
{
  return (phi[x+1][y][z] - phi[x-1][y][z]) / (2.0*dx);
}


double GradientY(double phi[][DATAYSIZE][DATAXSIZE], 
                 double dx, double dy, double dz, int x, int y, int z)
{
  return (phi[x][y+1][z] - phi[x][y-1][z]) / (2.0*dy);
}


double GradientZ(double phi[][DATAYSIZE][DATAXSIZE], 
                 double dx, double dy, double dz, int x, int y, int z)
{
  return (phi[x][y][z+1] - phi[x][y][z-1]) / (2.0*dz);
}


double Divergence(double phix[][DATAYSIZE][DATAXSIZE], 
                  double phiy[][DATAYSIZE][DATAXSIZE],
                  double phiz[][DATAYSIZE][DATAXSIZE], 
                  double dx, double dy, double dz, int x, int y, int z)
{
  return GradientX(phix,dx,dy,dz,x,y,z) + 
         GradientY(phiy,dx,dy,dz,x,y,z) +
         GradientZ(phiz,dx,dy,dz,x,y,z);
}


double Laplacian(double phi[][DATAYSIZE][DATAXSIZE],
                 double dx, double dy, double dz, int x, int y, int z)
{
  double phixx = (phi[x+1][y][z] + phi[x-1][y][z] - 2.0 * phi[x][y][z]) / SQ(dx);
  double phiyy = (phi[x][y+1][z] + phi[x][y-1][z] - 2.0 * phi[x][y][z]) / SQ(dy);
  double phizz = (phi[x][y][z+1] + phi[x][y][z-1] - 2.0 * phi[x][y][z]) / SQ(dz);
  return phixx + phiyy + phizz;
}


double An(double phix, double phiy, double phiz, double epsilon)
{
  if (phix != 0.0 || phiy != 0.0 || phiz != 0.0){
    return ((1.0 - 3.0 * epsilon) * (1.0 + (((4.0 * epsilon) / (1.0-3.0*epsilon))*
           ((SQ(phix)*SQ(phix)+SQ(phiy)*SQ(phiy)+SQ(phiz)*SQ(phiz)) /
           ((SQ(phix)+SQ(phiy)+SQ(phiz))*(SQ(phix)+SQ(phiy)+SQ(phiz)))))));
  }
  else
  {
    return (1.0-((5.0/3.0)*epsilon));
  }
}


double Wn(double phix, double phiy, double phiz, double epsilon, double W0)
{
  return (W0*An(phix,phiy,phiz,epsilon));
}


double taun(double phix, double phiy, double phiz, double epsilon, double tau0)
{
  return tau0 * SQ(An(phix,phiy,phiz,epsilon));
}


double dFunc(double l, double m, double n)
{
  if (l != 0.0 || m != 0.0 || n != 0.0){
    return (((l*l*l*(SQ(m)+SQ(n)))-(l*(SQ(m)*SQ(m)+SQ(n)*SQ(n)))) /
            ((SQ(l)+SQ(m)+SQ(n))*(SQ(l)+SQ(m)+SQ(n))));
  }
  else
  {
    return 0.0;
  }
}

void calculateForce(double phi[][DATAYSIZE][DATAXSIZE], 
                    double Fx[][DATAYSIZE][DATAXSIZE],
                    double Fy[][DATAYSIZE][DATAXSIZE],
                    double Fz[][DATAYSIZE][DATAXSIZE],
                    double dx, double dy, double dz,
                    double epsilon, double W0, double tau0,
                    nd_item<3> &item)
{

  unsigned iz = item.get_global_id(2);
  unsigned iy = item.get_global_id(1);
  unsigned ix = item.get_global_id(0);

  if ((ix < (DATAXSIZE-1)) && (iy < (DATAYSIZE-1)) && 
      (iz < (DATAZSIZE-1)) && (ix > (0)) && 
      (iy > (0)) && (iz > (0))) {

    double phix = GradientX(phi,dx,dy,dz,ix,iy,iz);
    double phiy = GradientY(phi,dx,dy,dz,ix,iy,iz);
    double phiz = GradientZ(phi,dx,dy,dz,ix,iy,iz);
    double sqGphi = SQ(phix) + SQ(phiy) + SQ(phiz);
    double c = 16.0 * W0 * epsilon;
    double w = Wn(phix,phiy,phiz,epsilon,W0);
    double w2 = SQ(w);
    

    Fx[ix][iy][iz] = w2 * phix + sqGphi * w * c * dFunc(phix,phiy,phiz);
    Fy[ix][iy][iz] = w2 * phiy + sqGphi * w * c * dFunc(phiy,phiz,phix);
    Fz[ix][iy][iz] = w2 * phiz + sqGphi * w * c * dFunc(phiz,phix,phiy);
  }
  else
  {
    Fx[ix][iy][iz] = 0.0;
    Fy[ix][iy][iz] = 0.0;
    Fz[ix][iy][iz] = 0.0;
  }

}

// device function to set the 3D volume
void allenCahn(double phinew[][DATAYSIZE][DATAXSIZE], 
               double phiold[][DATAYSIZE][DATAXSIZE],
               double uold[][DATAYSIZE][DATAXSIZE],
               double Fx[][DATAYSIZE][DATAXSIZE],
               double Fy[][DATAYSIZE][DATAXSIZE],
               double Fz[][DATAYSIZE][DATAXSIZE],
               double epsilon, double W0, double tau0, double lambda,
               double dt, double dx, double dy, double dz,
               nd_item<3> &item)
{
  unsigned iz = item.get_global_id(2);
  unsigned iy = item.get_global_id(1);
  unsigned ix = item.get_global_id(0);

  if ((ix < (DATAXSIZE-1)) && (iy < (DATAYSIZE-1)) && 
      (iz < (DATAZSIZE-1)) && (ix > (0)) && 
      (iy > (0)) && (iz > (0))) {

    double phix = GradientX(phiold,dx,dy,dz,ix,iy,iz);
    double phiy = GradientY(phiold,dx,dy,dz,ix,iy,iz);
    double phiz = GradientZ(phiold,dx,dy,dz,ix,iy,iz); 

    phinew[ix][iy][iz] = phiold[ix][iy][iz] + 
     (dt / taun(phix,phiy,phiz,epsilon,tau0)) * 
     (Divergence(Fx,Fy,Fz,dx,dy,dz,ix,iy,iz) - 
      dFphi(phiold[ix][iy][iz], uold[ix][iy][iz],lambda));
  }
}


void boundaryConditionsPhi(double phinew[][DATAYSIZE][DATAXSIZE],
                           nd_item<3> &item)
{
  unsigned iz = item.get_global_id(2);
  unsigned iy = item.get_global_id(1);
  unsigned ix = item.get_global_id(0);

  if (ix == 0){
    phinew[ix][iy][iz] = -1.0;
  }
  else if (ix == DATAXSIZE-1){
    phinew[ix][iy][iz] = -1.0;
  }
  else if (iy == 0){
    phinew[ix][iy][iz] = -1.0;
  }
  else if (iy == DATAYSIZE-1){
    phinew[ix][iy][iz] = -1.0;
  }
  else if (iz == 0){
    phinew[ix][iy][iz] = -1.0;
  }
  else if (iz == DATAZSIZE-1){
    phinew[ix][iy][iz] = -1.0;
  }
}


void thermalEquation(double unew[][DATAYSIZE][DATAXSIZE],
                     double uold[][DATAYSIZE][DATAXSIZE],
                     double phinew[][DATAYSIZE][DATAXSIZE],
                     double phiold[][DATAYSIZE][DATAXSIZE],
                     double D, double dt, double dx, double dy, double dz,
                     nd_item<3> &item)
{
  unsigned iz = item.get_global_id(2);
  unsigned iy = item.get_global_id(1);
  unsigned ix = item.get_global_id(0);

  if ((ix < (DATAXSIZE-1)) && (iy < (DATAYSIZE-1)) && 
      (iz < (DATAZSIZE-1)) && (ix > (0)) && 
      (iy > (0)) && (iz > (0))){
    unew[ix][iy][iz] = uold[ix][iy][iz] + 
      0.5*(phinew[ix][iy][iz]- phiold[ix][iy][iz]) +
      dt * D * Laplacian(uold,dx,dy,dz,ix,iy,iz);
  }
}


void boundaryConditionsU(double unew[][DATAYSIZE][DATAXSIZE], double delta,
                         nd_item<3> &item)
{
  unsigned iz = item.get_global_id(2);
  unsigned iy = item.get_global_id(1);
  unsigned ix = item.get_global_id(0);

  if (ix == 0){
    unew[ix][iy][iz] =  -delta;
  }
  else if (ix == DATAXSIZE-1){
    unew[ix][iy][iz] =  -delta;
  }
  else if (iy == 0){
    unew[ix][iy][iz] =  -delta;
  }
  else if (iy == DATAYSIZE-1){
    unew[ix][iy][iz] =  -delta;
  }
  else if (iz == 0){
    unew[ix][iy][iz] =  -delta;
  }
  else if (iz == DATAZSIZE-1){
    unew[ix][iy][iz] =  -delta;
  }
}


void swapGrid(double cnew[][DATAYSIZE][DATAXSIZE],
              double cold[][DATAYSIZE][DATAXSIZE],
              nd_item<3> &item)
{
  unsigned iz = item.get_global_id(2);
  unsigned iy = item.get_global_id(1);
  unsigned ix = item.get_global_id(0);

  if ((ix < (DATAXSIZE)) && 
      (iy < (DATAYSIZE)) &&
      (iz < (DATAZSIZE))) {
    double tmp = cnew[ix][iy][iz];
    cnew[ix][iy][iz] = cold[ix][iy][iz];
    cold[ix][iy][iz] = tmp;
  }
}

void initializationPhi(double phi[][DATAYSIZE][DATAXSIZE], double r0)
{
#ifdef _OPENMP
  #pragma omp parallel for collapse(3)
#endif
  for (int idx = 0; idx < DATAXSIZE; idx++) {
    for (int idy = 0; idy < DATAYSIZE; idy++) {
      for (int idz = 0; idz < DATAZSIZE; idz++) {
        double r = std::sqrt(SQ(idx-0.5*DATAXSIZE) + SQ(idy-0.5*DATAYSIZE) + SQ(idz-0.5*DATAZSIZE));
        if (r < r0){
          phi[idx][idy][idz] = 1.0;
        }
        else
        {
          phi[idx][idy][idz] = -1.0;
        }
      }
    }
  }
}

void initializationU(double u[][DATAYSIZE][DATAXSIZE], double r0, double delta)
{
#ifdef _OPENMP
  #pragma omp parallel for collapse(3)
#endif
  for (int idx = 0; idx < DATAXSIZE; idx++) {
    for (int idy = 0; idy < DATAYSIZE; idy++) {
      for (int idz = 0; idz < DATAZSIZE; idz++) {
        double r = std::sqrt(SQ(idx-0.5*DATAXSIZE) + SQ(idy-0.5*DATAYSIZE) + SQ(idz-0.5*DATAZSIZE));
        if (r < r0) {
          u[idx][idy][idz] = 0.0;
        }
        else
        {
          u[idx][idy][idz] = -delta * (1.0 - std::exp(-(r-r0)));
        }
      }
    }
  }
}

int main(int argc, char *argv[])
{
  const int num_steps = atoi(argv[1]);  //6000;
  const double dx = 0.4;
  const double dy = 0.4;
  const double dz = 0.4;
  const double dt = 0.01;
  const double delta = 0.8;
  const double r0 = 5.0;
  const double epsilon = 0.07;
  const double W0 = 1.0;
  const double beta0 = 0.0;
  const double D = 2.0;
  const double d0 = 0.5;
  const double a1 = 1.25 / std::sqrt(2.0);
  const double a2 = 0.64;
  const double lambda = (W0*a1)/(d0);
  const double tau0 = ((W0*W0*W0*a1*a2)/(d0*D)) + ((W0*W0*beta0)/(d0));

  // overall data set sizes
  const int nx = DATAXSIZE;
  const int ny = DATAYSIZE;
  const int nz = DATAZSIZE;
  const int vol = nx * ny * nz;

  // pointers for data set storage via malloc
  nRarray *phi_host;
  nRarray *u_host;

  phi_host = (nRarray *)malloc(vol*sizeof(double));
  u_host = (nRarray *)malloc(vol*sizeof(double));

  initializationPhi(phi_host,r0);
  initializationU(u_host,r0,delta);

#ifdef VERIFY
  nRarray *phi_ref = (nRarray *)malloc(vol*sizeof(double));
  nRarray *u_ref = (nRarray *)malloc(vol*sizeof(double));
  memcpy(phi_ref, phi_host, vol*sizeof(double));
  memcpy(u_ref, u_host, vol*sizeof(double));
  reference(phi_ref, u_ref, vol, num_steps);
#endif 

  auto offload_start = std::chrono::steady_clock::now();

  { // sycl scope
  #ifdef USE_GPU
    gpu_selector dev_sel;
  #else
    cpu_selector dev_sel;
  #endif
    queue q(dev_sel);

    // allocate GPU device buffers
    buffer<double, 1> d_phiold ((double*)phi_host, vol);
    buffer<double, 1> d_uold ((double*)u_host, vol);
    buffer<double, 1> d_phinew (vol);
    buffer<double, 1> d_unew (vol);
    buffer<double, 1> d_Fx (vol);
    buffer<double, 1> d_Fy (vol);
    buffer<double, 1> d_Fz (vol);

    // define the chunk sizes that each threadblock will work on
    range<3> gws ((DATAXSIZE+3)/4*4, (DATAYSIZE+7)/8*8, (DATAZSIZE+7)/8*8);
    range<3> lws (4, 8, 8);

    int t = 0;

    q.wait();
    auto start = std::chrono::steady_clock::now();

    auto d_phiold_re = d_phiold.reinterpret<nRarray>(range<1>(DATAZSIZE));
    auto d_phinew_re = d_phinew.reinterpret<nRarray>(range<1>(DATAZSIZE));
    auto d_uold_re = d_uold.reinterpret<nRarray>(range<1>(DATAZSIZE));
    auto d_unew_re = d_unew.reinterpret<nRarray>(range<1>(DATAZSIZE));
    auto d_Fx_re = d_Fx.reinterpret<nRarray>(range<1>(DATAZSIZE));
    auto d_Fy_re = d_Fy.reinterpret<nRarray>(range<1>(DATAZSIZE));
    auto d_Fz_re = d_Fz.reinterpret<nRarray>(range<1>(DATAZSIZE));

    while (t <= num_steps) {
      
      q.submit([&] (handler &cgh) {
        auto d_phiold = d_phiold_re.get_access<sycl_read>(cgh);
        auto d_Fx = d_Fx_re.get_access<sycl_discard_write>(cgh);
        auto d_Fy = d_Fy_re.get_access<sycl_discard_write>(cgh);
        auto d_Fz = d_Fz_re.get_access<sycl_discard_write>(cgh);
        cgh.parallel_for<class calc_force>(nd_range<3>(gws, lws), [=] (nd_item<3> item) {
          calculateForce(d_phiold.get_pointer(),
                         d_Fx.get_pointer(),
                         d_Fy.get_pointer(),
                         d_Fz.get_pointer(),
                         dx,dy,dz,epsilon,W0,tau0,
                         item);
        });
      });

      q.submit([&] (handler &cgh) {
        auto d_phinew = d_phinew_re.get_access<sycl_discard_write>(cgh);
        auto d_phiold = d_phiold_re.get_access<sycl_read>(cgh);
        auto d_uold = d_uold_re.get_access<sycl_read>(cgh);
        auto d_Fx = d_Fx_re.get_access<sycl_read>(cgh);
        auto d_Fy = d_Fy_re.get_access<sycl_read>(cgh);
        auto d_Fz = d_Fz_re.get_access<sycl_read>(cgh);
        cgh.parallel_for<class allen_cahn>(nd_range<3>(gws, lws), [=] (nd_item<3> item) {
          allenCahn(d_phinew.get_pointer(),
                    d_phiold.get_pointer(),
                    d_uold.get_pointer(),
                    d_Fx.get_pointer(),
                    d_Fy.get_pointer(),
                    d_Fz.get_pointer(),
                    epsilon,W0,tau0,lambda,
                    dt,dx,dy,dz,
                    item);
        });
      });

      q.submit([&] (handler &cgh) {
        auto d_phinew = d_phinew_re.get_access<sycl_write>(cgh);
        cgh.parallel_for<class bc_phi>(nd_range<3>(gws, lws), [=] (nd_item<3> item) {
          boundaryConditionsPhi(d_phinew.get_pointer(), item);
        });
      });

      q.submit([&] (handler &cgh) {
        auto d_unew = d_unew_re.get_access<sycl_discard_write>(cgh);
        auto d_uold = d_uold_re.get_access<sycl_read>(cgh);
        auto d_phinew = d_phinew_re.get_access<sycl_read>(cgh);
        auto d_phiold = d_phiold_re.get_access<sycl_read>(cgh);
        cgh.parallel_for<class thermal_equation>(nd_range<3>(gws, lws), [=] (nd_item<3> item) {
          thermalEquation(d_unew.get_pointer(),
                          d_uold.get_pointer(),
                          d_phinew.get_pointer(),
                          d_phiold.get_pointer(),
                          D,dt,dx,dy,dz,
                          item);
        });
      });

      q.submit([&] (handler &cgh) {
        auto d_unew = d_unew_re.get_access<sycl_write>(cgh);
        cgh.parallel_for<class bc_u>(nd_range<3>(gws, lws), [=] (nd_item<3> item) {
          boundaryConditionsU(d_unew.get_pointer(), delta, item);
        });
      });

      q.submit([&] (handler &cgh) {
        auto d_phinew = d_phinew_re.get_access<sycl_read_write>(cgh);
        auto d_phiold = d_phiold_re.get_access<sycl_read_write>(cgh);
        cgh.parallel_for<class swap_phi>(nd_range<3>(gws, lws), [=] (nd_item<3> item) {
          swapGrid(d_phinew.get_pointer(), d_phiold.get_pointer(), item);
        });
      });

      q.submit([&] (handler &cgh) {
        auto d_unew = d_unew_re.get_access<sycl_read_write>(cgh);
        auto d_uold = d_uold_re.get_access<sycl_read_write>(cgh);
        cgh.parallel_for<class swap_u>(nd_range<3>(gws, lws), [=] (nd_item<3> item) {
          swapGrid(d_unew.get_pointer(), d_uold.get_pointer(), item);
        });
      });

      t++;
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Total kernel execution time: %.3f (ms)\n", time * 1e-6f);

  } // sycl scope

  auto offload_end = std::chrono::steady_clock::now();
  auto offload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(offload_end - offload_start).count();
  printf("Offload time: %.3f (ms)\n", offload_time * 1e-6f);

#ifdef VERIFY
  bool ok = true;
  for (int idx = 0; idx < nx; idx++)
    for (int idy = 0; idy < ny; idy++)
      for (int idz = 0; idz < nz; idz++) {
        if (fabs(phi_ref[idx][idy][idz] - phi_host[idx][idy][idz]) > 1e-3) {
          ok = false; printf("phi: %lf %lf\n", phi_ref[idx][idy][idz], phi_host[idx][idy][idz]);
	}
        if (fabs(u_ref[idx][idy][idz] - u_host[idx][idy][idz]) > 1e-3) {
          ok = false; printf("u: %lf %lf\n", u_ref[idx][idy][idz], u_host[idx][idy][idz]);
        }
      }
  printf("%s\n", ok ? "PASS" : "FAIL");
  free(phi_ref);
  free(u_ref);
#endif

  free(phi_host);
  free(u_host);

  return 0;
}
