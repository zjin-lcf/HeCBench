#include <iostream>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

int main(int argc, char * argv[])
{
  if (argc != 4) {
    std::cout << "Usage ./main <iterations> <width> <height>\n";
    std::cout << "Width and height are a multiple of 256\n";
    return 1;
  }
  int iterations = atoi(argv[1]);  // Simulation iterations
  int lbm_width = atoi(argv[2]);   // Dimension of LBM simulation area
  int lbm_height = atoi(argv[3]);

  int dims[2] = {lbm_width, lbm_height};
  size_t temp = dims[0] * dims[1];

   // Directions
   double e[9][2] = {{0,0}, {1,0}, {0,1}, {-1,0}, {0,-1}, {1,1}, {-1,1}, {-1,-1}, {1,-1}};
   
   // Weights
   double w[9] = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};

  // Omega
  const double omega = 1.2f;

  double8 dirX, dirY; // Directions

  // host inputs
  double *h_if0 = (double*)malloc(sizeof(double) * temp);
  double *h_if1234 = (double*)malloc(sizeof(double4) * temp);
  double *h_if5678 = (double*)malloc(sizeof(double4) * temp);

#ifdef VERIFY
  // Reference outputs
  double *v_of0 = (double*)malloc(sizeof(double) * temp);
  double *v_of1234 = (double*)malloc(sizeof(double4) * temp);
  double *v_of5678 = (double*)malloc(sizeof(double4) * temp);
#endif

  // Host outputs
  double *h_of0 = (double*)malloc(sizeof(double) * temp);
  double *h_of1234 = (double*)malloc(sizeof(double4) * temp);
  double *h_of5678 = (double*)malloc(sizeof(double4) * temp);

  // Cell Type - Boundary = 1 or Fluid = 0
  bool *h_type = (bool*)malloc(sizeof(bool) * temp);
  
  // Density
  double *rho = (double*)malloc(sizeof(double) * temp);
  
  // Velocity
  double2 *u = (double2*)malloc(sizeof(double2) * temp);

  // Initial velocity is nonzero for verifying host and device results
  double u0[2] = {0.01, 0.01};

  srand(123);
  for (int y = 0; y < dims[1]; y++)
  {
    for (int x = 0; x < dims[0]; x++)
    {
      int pos = x + y * dims[0];

      // Random values for verification
      double den = rand() % 10 + 1;

      // Initialize the velocity buffer
      u[pos] = double2(u0[0], u0[1]);

      h_if0[pos] = computefEq(w[0], e[0], den, u0);
      h_if1234[pos * 4 + 0] = computefEq(w[1], e[1], den, u0);
      h_if1234[pos * 4 + 1] = computefEq(w[2], e[2], den, u0);
      h_if1234[pos * 4 + 2] = computefEq(w[3], e[3], den, u0);
      h_if1234[pos * 4 + 3] = computefEq(w[4], e[4], den, u0);
      h_if5678[pos * 4 + 0] = computefEq(w[5], e[5], den, u0);
      h_if5678[pos * 4 + 1] = computefEq(w[6], e[6], den, u0);
      h_if5678[pos * 4 + 2] = computefEq(w[7], e[7], den, u0);
      h_if5678[pos * 4 + 3] = computefEq(w[8], e[8], den, u0);

      // Initialize boundary cells
      if (x == 0 || x == (dims[0] - 1) || y == 0 || y == (dims[1] - 1))
        h_type[pos] = 1;

      // Initialize fluid cells
      else
        h_type[pos] = 0;
    }
  }

  // initialize direction vectors
  dirX.s0() = e[1][0]; dirY.s0() = e[1][1];
  dirX.s1() = e[2][0]; dirY.s1() = e[2][1];
  dirX.s2() = e[3][0]; dirY.s2() = e[3][1];
  dirX.s3() = e[4][0]; dirY.s3() = e[4][1];
  dirX.s4() = e[5][0]; dirY.s4() = e[5][1];
  dirX.s5() = e[6][0]; dirY.s5() = e[6][1];
  dirX.s6() = e[7][0]; dirY.s6() = e[7][1];
  dirX.s7() = e[8][0]; dirY.s7() = e[8][1];

#ifdef VERIFY
  reference (iterations, omega, dims, h_type, rho, e, w,
             h_if0, h_if1234, h_if5678,
             v_of0, v_of1234, v_of5678);
#endif

  fluidSim (iterations, omega, dims, h_type, u, rho, dirX, dirY, w,
             h_if0, h_if1234, h_if5678,
             h_of0, h_of1234, h_of5678);

#ifdef VERIFY
  verify(dims, h_of0, h_of1234, h_of5678,
               v_of0, v_of1234, v_of5678);

  if(v_of0)    free(v_of0);
  if(v_of1234) free(v_of1234);
  if(v_of5678) free(v_of5678);
#endif
  
  if(h_if0)    free(h_if0);
  if(h_if1234) free(h_if1234);
  if(h_if5678) free(h_if5678);
  if(h_of0)    free(h_of0);
  if(h_of1234) free(h_of1234);
  if(h_of5678) free(h_of5678);
  if(h_type)   free(h_type);
  if(rho)      free(rho);
  if(u)        free(u);

  return 0;
}
