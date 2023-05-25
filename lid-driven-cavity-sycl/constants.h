#include <math.h>

#ifndef CONST_H
#define CONST_H

/** Problem size along one side; total number of cells is this squared */
#define NUM 512

// block size
#define BLOCK_SIZE 128

/** Double precision */
#define DOUBLE

#ifdef DOUBLE

  #define Real double
  
  #define FMIN std::fmin
  #define FMAX std::fmax
  #define FABS std::fabs
  #define SQRT std::sqrt
  
  #define ZERO 0.0
  #define ONE 1.0
  #define TWO 2.0
  #define FOUR 4.0
  
  #define SMALL 1.0e-10;
  
  /** Reynolds number */
  const Real Re_num = 1000.0;
  
  /** SOR relaxation parameter */
  const Real omega = 1.7;
  
  /** Discretization mixture parameter (gamma) */
  const Real mix_param = 0.9;
  
  /** Safety factor for time step modification */
  const Real tau = 0.5;
  
  /** Body forces in x- and y- directions */
  const Real gx = 0.0;
  const Real gy = 0.0;
  
  /** Domain size (non-dimensional) */
  #define xLength 1.0
  #define yLength 1.0

#else

  #define Real float

  #define ZERO 0.0f
  #define ONE 1.0f
  #define TWO 2.0f
  #define FOUR 4.0f
  #define SMALL 1.0e-10f;
  
  #define FMIN std::fminf
  #define FMAX std::fmaxf
  #define FABS std::fabsf
  #define SQRT std::sqrtf
  /** Reynolds number */
  const Real Re_num = 1000.0f;
  
  /** SOR relaxation parameter */
  const Real omega = 1.7f;
  
  /** Discretization mixture parameter (gamma) */
  const Real mix_param = 0.9f;
  
  /** Safety factor for time step modification */
  const Real tau = 0.5f;
  
  /** Body forces in x- and y- directions */
  const Real gx = 0.0f;
  const Real gy = 0.0f;
  
  /** Domain size (non-dimensional) */
  #define xLength 1.0f
  #define yLength 1.0f

#endif

#endif
