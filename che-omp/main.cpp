#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <random>
#include <fstream>
#include <omp.h>

//define the data set size for a cubic volume
#define DATAXSIZE 256
#define DATAYSIZE 256
#define DATAZSIZE 256

//define the chunk sizes that each threadblock will work on
#define BLKXSIZE 16
#define BLKYSIZE 4
#define BLKZSIZE 4

using namespace std;

#pragma omp declare target
double Laplacian(const double c[][DATAYSIZE][DATAXSIZE],
                 double dx, double dy, double dz, int x, int y, int z)
{
  int xp, xn, yp, yn, zp, zn;

  int nx = (int)DATAXSIZE - 1;
  int ny = (int)DATAYSIZE - 1;
  int nz = (int)DATAZSIZE - 1;

  xp = x+1;
  xn = x-1;
  yp = y+1;
  yn = y-1;
  zp = z+1;
  zn = z-1;

  if (xp > nx) xp = 0;
  if (yp > ny) yp = 0;
  if (zp > nz) zp = 0;
  if (xn < 0)  xn = nx;
  if (yn < 0)  yn = ny;
  if (zn < 0)  zn = nz;

  double cxx = (c[z][y][xp] + c[z][y][xn] - 2.0*c[z][y][x]) / (dx*dx);
  double cyy = (c[z][yp][x] + c[z][yn][x] - 2.0*c[z][y][x]) / (dy*dy);
  double czz = (c[zp][y][x] + c[zn][y][x] - 2.0*c[z][y][x]) / (dz*dz);

  return cxx + cyy + czz;
}

double GradientX(const double phi[][DATAYSIZE][DATAXSIZE], 
                 double dx, double dy, double dz, int x, int y, int z)
{
  int nx = (int)DATAXSIZE - 1;
  int xp = x+1;
  int xn = x-1;

  if (xp > nx) xp = 0;
  if (xn < 0)  xn = nx;

  return (phi[z][y][xp] - phi[z][y][xn]) / (2.0*dx);
}

double GradientY(const double phi[][DATAYSIZE][DATAXSIZE], 
                 double dx, double dy, double dz, int x, int y, int z)
{
  int ny = (int)DATAYSIZE - 1;
  int yp = y+1;
  int yn = y-1;

  if (yp > ny) yp = 0;
  if (yn < 0)  yn = ny;

  return (phi[z][yp][x] - phi[z][yn][x]) / (2.0*dy);
}

double GradientZ(const double phi[][DATAYSIZE][DATAXSIZE],
                 double dx, double dy, double dz, int x, int y, int z)
{
  int nz = (int)DATAZSIZE - 1;
  int zp = z+1;
  int zn = z-1;

  if (zp > nz) zp = 0;
  if (zn < 0)  zn = nz;

  return (phi[zp][y][x] - phi[zn][y][x]) / (2.0*dz);
}

double freeEnergy(double c, double e_AA, double e_BB, double e_AB)
{
  return (((9.0 / 4.0) * ((c*c+2.0*c+1.0)*e_AA+(c*c-2.0*c+1.0)*e_BB+
          2.0*(1.0-c*c)*e_AB)) + ((3.0/2.0) * c * c) + ((3.0/12.0) * c * c * c * c));
}

#pragma omp end declare target

void chemicalPotential(
    const double c[][DATAYSIZE][DATAXSIZE], 
    double mu[][DATAYSIZE][DATAXSIZE], 
    double dx,
    double dy,
    double dz,
    double gamma,
    double e_AA,
    double e_BB,
    double e_AB)
{
  #pragma omp target teams distribute parallel for collapse(3)
  for (int idz = 0; idz < DATAZSIZE; idz++) {
    for (int idy = 0; idy < DATAYSIZE; idy++) {
      for (int idx = 0; idx < DATAXSIZE; idx++) {

        mu[idz][idy][idx] = 4.5 * ( ( c[idz][idy][idx] + 1.0 ) * e_AA + 
            ( c[idz][idy][idx] - 1 ) * e_BB - 2.0 * c[idz][idy][idx] * e_AB ) + 
          3.0 * c[idz][idy][idx] + c[idz][idy][idx] * c[idz][idy][idx] * c[idz][idy][idx] - 
          gamma * Laplacian(c,dx,dy,dz,idx,idy,idz);
      }
    }
  }
}

void localFreeEnergyFunctional(
    const double c[][DATAYSIZE][DATAXSIZE],
    double f[][DATAYSIZE][DATAXSIZE], 
    double dx,
    double dy,
    double dz,
    double gamma,
    double e_AA,
    double e_BB,
    double e_AB)
{
  #pragma omp target teams distribute parallel for collapse(3)
  for (int idz = 0; idz < DATAZSIZE; idz++) {
    for (int idy = 0; idy < DATAYSIZE; idy++) {
      for (int idx = 0; idx < DATAXSIZE; idx++) {

        f[idz][idy][idx] = freeEnergy(c[idz][idy][idx],e_AA,e_BB,e_AB) + (gamma / 2.0) * (
            GradientX(c,dx,dy,dz,idx,idy,idz) * GradientX(c,dx,dy,dz,idx,idy,idz) + 
            GradientY(c,dx,dy,dz,idx,idy,idz) * GradientY(c,dx,dy,dz,idx,idy,idz) + 
            GradientZ(c,dx,dy,dz,idx,idy,idz) * GradientZ(c,dx,dy,dz,idx,idy,idz));
      }
    }
  }
}

void cahnHilliard(
    double cnew[][DATAYSIZE][DATAXSIZE], 
    const double cold[][DATAYSIZE][DATAXSIZE], 
    const double mu[][DATAYSIZE][DATAXSIZE],
    double D,
    double dt,
    double dx,
    double dy,
    double dz)
{
  #pragma omp target teams distribute parallel for collapse(3)
  for (int idz = 0; idz < DATAZSIZE; idz++) {
    for (int idy = 0; idy < DATAYSIZE; idy++) {
      for (int idx = 0; idx < DATAXSIZE; idx++) {
        cnew[idz][idy][idx] = cold[idz][idy][idx] + dt * D * Laplacian(mu,dx,dy,dz,idx,idy,idz);
      }
    }
  }
}

void Swap(double cnew[][DATAYSIZE][DATAXSIZE], double cold[][DATAYSIZE][DATAXSIZE])
{
  #pragma omp target teams distribute parallel for collapse(3)
  for (int idz = 0; idz < DATAZSIZE; idz++) {
    for (int idy = 0; idy < DATAYSIZE; idy++) {
      for (int idx = 0; idx < DATAXSIZE; idx++) {
        double tmp = cnew[idz][idy][idx];
        cnew[idz][idy][idx] = cold[idz][idy][idx];
        cold[idz][idy][idx] = tmp;
      }
    }
  }

}

void initialization(double c[][DATAYSIZE][DATAXSIZE])
{
  srand(2);
  for (unsigned int idz = 0.0; idz < DATAZSIZE; idz++) {
    for (unsigned int idy = 0.0; idy < DATAYSIZE; idy++) {
      for (unsigned int idx = 0.0; idx < DATAXSIZE; idx++) {
        double f = (double)rand() / RAND_MAX;
        c[idz][idy][idx] = -1.0 + 2.0*f;
      }
    }
  }
}

double integral(const double c[][DATAYSIZE][DATAXSIZE], int nx, int ny, int nz)
{
  double summation = 0.0;  

  for (int k = 0; k < nz; k++)
    for(int j = 0; j < ny; j++)
      for(int i = 0; i < nx; i++)
        summation = summation + c[k][j][i];

  return summation;
}

int main(int argc, char *argv[])
{
  const double dx = 1.0;
  const double dy = 1.0;
  const double dz = 1.0;
  const double dt = 0.01;
  const double e_AA = -(2.0/9.0);
  const double e_BB = -(2.0/9.0);
  const double e_AB = (2.0/9.0);
  const int t_f = atoi(argv[1]);    // default value: 25000
#ifndef DEBUG
  const int t_freq = t_f; 
#else
  const int t_freq = 10;
#endif
  const double gamma = 0.5;
  const double D = 1.0;

  string name_c = "./out/integral_c.txt";
  ofstream ofile_c (name_c);

  string name_mu = "./out/integral_mu.txt";
  ofstream ofile_mu (name_mu);

  string name_f = "./out/integral_f.txt";
  ofstream ofile_f (name_f);

  typedef double nRarray[DATAYSIZE][DATAXSIZE];

  // overall data set sizes
  const int nx = DATAXSIZE;
  const int ny = DATAYSIZE;
  const int nz = DATAZSIZE;
  const int vol = nx * ny * nz;
  const size_t vol_bytes = vol * sizeof(double);

  // pointers for data set storage via malloc
  nRarray *cold; // storage for result stored on host
  nRarray *cnew; // storage for result stored on host
  nRarray *muold;
  nRarray *fold;

  if ((cold = (nRarray *)malloc(vol_bytes)) == 0) {
    fprintf(stderr,"cold host malloc failed\n"); 
    return 1;
  }
  if ((cnew = (nRarray *)malloc(vol_bytes)) == 0) {
    fprintf(stderr,"cnew host malloc failed\n"); 
    return 1;
  }
  if ((muold = (nRarray *)malloc(vol_bytes)) == 0) {
    fprintf(stderr,"muold host malloc failed\n"); 
    return 1;
  }
  if ((fold = (nRarray *)malloc(vol_bytes)) == 0) {
    fprintf(stderr,"fold host malloc failed\n"); 
    return 1;
  }

  initialization(cold);

  // cast the nRarray type
  double *co  = (double*) cold;
  double *mu = (double*) muold;
  double *f  = (double*) fold;
  double *cn  = (double*) cnew;

#pragma omp target data map(to: co[0:vol]) \
                        map(alloc: cn[0:vol], mu[0:vol], f[0:vol])

{
  auto start = std::chrono::steady_clock::now();
  double integral_c = 0.0;
  double integral_mu = 0.0;
  double integral_f = 0.0;

  for (int t = 0; t < t_f; t++) {

    chemicalPotential(cold,muold,dx,dy,dz,gamma,e_AA,e_BB,e_AB);
    localFreeEnergyFunctional(cold,fold,dx,dy,dz,gamma,e_AA,e_BB,e_AB);
    cahnHilliard(cnew,cold,muold,D,dt,dx,dy,dz);

    if (t > 0 && t % (t_freq - 1) == 0) {
      #pragma omp target update from(cn[0:vol])
      #pragma omp target update from(mu[0:vol])
      #pragma omp target update from(f[0:vol])

      integral_c = integral(cnew,nx,ny,nz);

      ofile_c << t << "," << integral_c << endl;

      integral_mu = integral(muold,nx,ny,nz);

      ofile_mu << t << "," << integral_mu << endl;

      integral_f = integral(fold,nx,ny,nz);

      ofile_f << t << "," << integral_f << endl;
    }

    Swap(cnew, cold);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Kernel exeuction time on the GPU (%d iterations) = %.3f (s)\n", t_f, time * 1e-9f);
}

  free(cnew);
  free(cold);
  free(muold);
  free(fold);
  return 0;
}
