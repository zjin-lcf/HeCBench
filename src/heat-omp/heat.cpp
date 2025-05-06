/*
** PROGRAM: heat equation solve
**
** PURPOSE: This program will explore use of an explicit
**          finite difference method to solve the heat
**          equation under a method of manufactured solution (MMS)
**          scheme. The solution has been set to be a simple 
**          function based on exponentials and trig functions.
**
**          A finite difference scheme is used on a 1000x1000 cube.
**          A total of 0.5 units of time are simulated.
**
**          The MMS solution has been adapted from
**          G.W. Recktenwald (2011). Finite difference approximations
**          to the Heat Equation. Portland State University.
**
**
** USAGE:   Run with two arguments:
**          First is the number of cells.
**          Second is the number of timesteps.
**
**          For example, with 100x100 cells and 10 steps:
**
**          ./heat 100 10
**
**
** HISTORY: Written by Tom Deakin, Oct 2018
**
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Key constants used in this program
#define LINE "--------------------\n" // A line for fancy output

double solution(const double t, const double x, const double y, const double alpha, const double length);
double l2norm(const int n, const double * __restrict u, const int nsteps, const double dt, const double alpha, const double dx, const double length);

// begin of initial_value
void initial_value(const int numTeams,
                   const int numThreads,
                   const unsigned int n,
                   const double dx,
                   const double length,
                   double * u)
{
  // Sets the mesh to an initial value, determined by the MMS scheme
  #pragma omp target teams distribute parallel for collapse(2) \
   num_teams(numTeams) num_threads(numThreads)
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      double y = (j+1)*dx; // Physical y position
      double x = (i+1)*dx; // Physical x position
      u[i+j*n] = sin(acos(-1.0) * x / length) * sin(acos(-1.0) * y / length);
    }
  }
}
// end of initial_value

void zero(const int numTeams,
          const int numThreads,
          const unsigned int n, double * u)
{
  #pragma omp target teams distribute parallel for \
   num_teams(numTeams) num_threads(numThreads)
  for (int i = 0; i < n * n; ++i) {
    u[i] = 0.0;
  }
}

void solve(const int numTeams,
           const int numThreads,
           const unsigned int n,
           const double alpha,
           const double dx,
           const double dt, 
           const double r,
           const double r2,
           const double * __restrict u,
           double * __restrict u_tmp)
{
  #pragma omp target teams distribute parallel for collapse(2) \
   num_teams(numTeams) num_threads(numThreads)
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      // Update the 5-point stencil, using boundary conditions on the edges of the domain.
      // Boundaries are zero because the MMS solution is zero there.
      u_tmp[i+j*n] =  r2 * u[i+j*n] +
      r * ((i < n-1) ? u[i+1+j*n] : 0.0) +
      r * ((i > 0)   ? u[i-1+j*n] : 0.0) +
      r * ((j < n-1) ? u[i+(j+1)*n] : 0.0) +
      r * ((j > 0)   ? u[i+(j-1)*n] : 0.0);
    }
  }
}

int main(int argc, char *argv[]) {

  // Start the total program runtime timer
  double start = omp_get_wtime();

  // Problem size, forms an nxn grid
  int n = 1000;

  // Number of timesteps
  int nsteps = 10;

  // Check for the correct number of arguments
  // Print usage and exits if not correct
  if (argc == 3) {

    // Set problem size from first argument
    n = atoi(argv[1]);
    if (n < 0) {
      fprintf(stderr, "Error: n must be positive\n");
      exit(EXIT_FAILURE);
    }

    // Set number of timesteps from second argument
    nsteps = atoi(argv[2]);
    if (nsteps < 0) {
      fprintf(stderr, "Error: nsteps must be positive\n");
      exit(EXIT_FAILURE);
    }
  }

  //
  // Set problem definition
  //
  double alpha = 0.1;          // heat equation coefficient
  double length = 1000.0;      // physical size of domain: length x length square
  double dx = length / (n+1);  // physical size of each cell (+1 as don't simulate boundaries as they are given)
  double dt = 0.5 / nsteps;    // time interval (total time of 0.5s)

  // Stability requires that dt/(dx^2) <= 0.5,
  double r = alpha * dt / (dx * dx);

  // Print message detailing runtime configuration
  printf("\n");
  printf(" MMS heat equation\n\n");
  printf(LINE);
  printf("Problem input\n\n");
  printf(" Grid size: %d x %d\n", n, n);
  printf(" Cell width: %E\n", dx);
  printf(" Grid length: %lf x %lf\n", length, length);
  printf("\n");
  printf(" Alpha: %E\n", alpha);
  printf("\n");
  printf(" Steps: %d\n", nsteps);
  printf(" Total time: %E\n", dt*(double)nsteps);
  printf(" Time step: %E\n", dt);
  printf(LINE);

  // Stability check
  printf("Stability\n\n");
  printf(" r value: %lf\n", r);
  if (r > 0.5)
    printf(" Warning: unstable\n");
  printf(LINE);

  // Allocate two nxn grids
  double *u     = (double*) malloc(sizeof(double)*n*n);
  double *u_tmp = (double*) malloc(sizeof(double)*n*n);

  double tic, toc;
  const int block_size = 256;

  #pragma omp target data map(tofrom: u[0:n*n], u_tmp[0:n*n]) 
  {
    int n_ceil = (n*n+block_size-1) / block_size;
  
    // Set the initial value of the grid under the MMS scheme
    initial_value(n_ceil, block_size, n, dx, length, u);
  
    zero(n_ceil, block_size, n, u_tmp);

    //
    // Run through timesteps under the explicit scheme
    //

    // Finite difference constant multiplier
    const double r2 = 1.0 - 4.0*r;

    // Start the solve timer
    tic = omp_get_wtime();

    for (int t = 0; t < nsteps; ++t) {

      // Call the solve kernel
      // Computes u_tmp at the next timestep
      // given the value of u at the current timestep
      // Loop over the nxn grid
      solve(n_ceil, block_size, n, alpha, dx, dt, r, r2, u, u_tmp);

      // Pointer swap
      double *tmp = u;
      u = u_tmp;
      u_tmp = tmp;
    }
    // Stop solve timer
    toc = omp_get_wtime();
  }

  //
  // Check the L2-norm of the computed solution
  // against the *known* solution from the MMS scheme
  //
  double norm = l2norm(n, u, nsteps, dt, alpha, dx, length);

  // Stop total timer
  double stop = omp_get_wtime();

  // Print results
  printf("Results\n\n");
  printf("Error (L2norm): %E\n", norm);
  printf("Solve time (s): %lf\n", toc-tic);
  printf("Total time (s): %lf\n", stop-start);
  printf("Bandwidth (GB/s): %lf\n", 1.0E-9*2.0*n*n*nsteps*sizeof(double)/(toc-tic));
  printf(LINE);

  // Free the memory
  free(u);
  free(u_tmp);
}


// True answer given by the manufactured solution
double solution(const double t, const double x, const double y, const double alpha, const double length) {

  return exp(-2.0*alpha*M_PI*M_PI*t/(length*length)) * sin(M_PI*x/length) * sin(M_PI*y/length);

}


// Computes the L2-norm of the computed grid and the MMS known solution
// The known solution is the same as the boundary function.
double l2norm(const int n, const double * u, const int nsteps, const double dt,
              const double alpha, const double dx, const double length) {

  // Final (real) time simulated
  double time = dt * (double)nsteps;

  // L2-norm error
  double l2norm = 0.0;

  // Loop over the grid and compute difference of computed and known solutions as an L2-norm
  double y = dx;
  for (int j = 0; j < n; ++j) {
    double x = dx;
    for (int i = 0; i < n; ++i) {
      double answer = solution(time, x, y, alpha, length);
      l2norm += (u[i+j*n] - answer) * (u[i+j*n] - answer);

      x += dx;
    }
    y += dx;
  }

  return sqrt(l2norm);
}
