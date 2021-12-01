#ifndef UTILS_H
#define UTILS_H

#include <omp.h>

typedef struct __attribute__((__aligned__(32)))
{
  int s0, s1, s2, s3, s4, s5, s6, s7;
}
int8;

typedef struct __attribute__((__aligned__(64)))
{
  double s0, s1, s2, s3, s4, s5, s6, s7;
}
double8;

typedef struct __attribute__((__aligned__(32)))
{
  double x, y, z, w;
}
double4;

typedef struct __attribute__((__aligned__(16)))
{
  double x, y;
}
double2;

void fluidSim (
  const int iterations,
  const double omega,
  const int *dims,
  const bool *h_type,
  double2 *u,
  double *rho,
  const double8 dirX,
  const double8 dirY,
  const double w[9],     // Weights
        double *h_if0,
        double *h_if1234,
        double *h_if5678,
        double *h_of0,
        double *h_of1234,
        double *h_of5678);

// Calculates equivalent distribution 
double computefEq(double rho, double weight, const double dir[2], const double velocity[2]);

#ifdef VERIFY
void reference (
  const int iterations,
  const double omega,
  const int *dims,
  const bool *h_type,
  double *rho,
  const double (*e)[2],  // Directions
  const double w[9],     // Weights
  const double *h_if0,
  const double *h_if1234,
  const double *h_if5678,
        double *v_of0,
        double *v_of1234,
        double *v_of5678);

void verify(
  const int *dims,
  const double *h_of0,
  const double *h_of1234,
  const double *h_of5678,
  const double *v_of0,
  const double *v_of1234,
  const double *v_of5678);
#endif

#endif
