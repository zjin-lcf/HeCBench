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
**          Ported to SYCL by Tom Deakin, Nov 2019
**
*/

#include <iostream>
#include <chrono>
#include <cmath>
#include <sycl/sycl.hpp>

// Key constants used in this program
#define PI sycl::acos(-1.0) // Pi
#define LINE "--------------------" // A line for fancy output

void initial_value(sycl::queue &q, const unsigned int n, const double dx, const double length, double *u);
void zero(sycl::queue &q, const unsigned int n, double *u);
void solve(sycl::queue &q, const unsigned int n, const double alpha, const double dx, const double dt, double *u, double *u_tmp);
double solution(const double t, const double x, const double y, const double alpha, const double length);
double l2norm(const unsigned int n, const double * u, const int nsteps, const double dt, const double alpha, const double dx, const double length);

// Main function
int main(int argc, char *argv[]) {

  // Start the total program runtime timer
  auto start = std::chrono::high_resolution_clock::now();

  // Problem size, forms an nxn grid
  unsigned int n = 1000;

  // Number of timesteps
  int nsteps = 10;

  // Check for the correct number of arguments
  // Print usage and exits if not correct
  if (argc == 3) {

    // Set problem size from first argument
    n = atoi(argv[1]);
    if (n < 0) {
      std::cerr << "Error: n must be positive" << std::endl;
      exit(EXIT_FAILURE);
    }

    // Set number of timesteps from second argument
    nsteps = atoi(argv[2]);
    if (nsteps < 0) {
      std::cerr << "Error: nsteps must be positive" << std::endl;
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

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // Print message detailing runtime configuration
  std::cout
    << std::endl
    << " MMS heat equation" << std::endl << std::endl
    << LINE << std::endl
    << "Problem input" << std::endl << std::endl
    << " Grid size: " << n << " x " << n << std::endl
    << " Cell width: " << dx << std::endl
    << " Grid length: " << length << "x" << length << std::endl
    << std::endl
    << " Alpha: " << alpha << std::endl
    << std::endl
    << " Steps: " <<  nsteps << std::endl
    << " Total time: " << dt*(double)nsteps << std::endl
    << " Time step: " << dt << std::endl
    << " GPU device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl
    << LINE << std::endl;

  // Stability check
  std::cout << "Stability" << std::endl << std::endl;
  std::cout << " r value: " << r << std::endl;
  if (r > 0.5)
    std::cout << " Warning: unstable" << std::endl;
  std::cout << LINE << std::endl;

  const unsigned int grid_size = n * n;

  // Allocate two nxn grids
  double *u = sycl::malloc_device<double>(grid_size, q);
  double *u_tmp = sycl::malloc_device<double>(grid_size, q);

  const int block_size = 256;
  const int n_ceil = (grid_size+block_size-1) / block_size * block_size;

  // Set the initial value of the grid under the MMS scheme
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class initial_value_kernel>(
      sycl::nd_range<1>(sycl::range<1>(n_ceil), sycl::range<1>(block_size)),
      [=](sycl::nd_item<1> item) {
      int idx = item.get_global_id(0);
      if (idx < grid_size) {
        int i = idx % n;
        int j = idx / n;
        double y = dx * (j+1); // Physical y position
        double x = dx * (i+1); // Physical x position
        u[i+j*n] = sycl::sin(PI * x / length) * sycl::sin(PI * y / length);
      }
    });
  });

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class zero_kernel>(
      sycl::nd_range<1>(sycl::range<1>(n_ceil), sycl::range<1>(block_size)),
      [=](sycl::nd_item<1> item) {
      int idx = item.get_global_id(0);
      if (idx < grid_size) u_tmp[idx] = 0.0;
    });
  });

  // Ensure everything is initalised on the device
  q.wait();

  //
  // Run through timesteps under the explicit scheme
  //

  const double r2 = 1.0 - 4.0*r;

  // Start the solve timer
  auto tic = std::chrono::high_resolution_clock::now();

  for (int t = 0; t < nsteps; ++t) {

    // Call the solve kernel
    // Computes u_tmp at the next timestep
    // given the value of u at the current timestep
    q.submit([&](sycl::handler& cgh) {
      // Loop over the nxn grid
      cgh.parallel_for<class solve_kernel>(
        sycl::nd_range<1>(sycl::range<1>(n_ceil), sycl::range<1>(block_size)),
        [=](sycl::nd_item<1> item) {
        int idx = item.get_global_id(0);
        if (idx < grid_size) {
          int i = idx % n;
          int j = idx / n;

          // Update the 5-point stencil, using boundary conditions on the edges of the domain.
          // Boundaries are zero because the MMS solution is zero there.
          u_tmp[i+j*n] = r2 * u[i+j*n] +
          r * ((i < n-1) ? u[i+1+j*n] : 0.0) +
          r * ((i > 0)   ? u[i-1+j*n] : 0.0) +
          r * ((j < n-1) ? u[i+(j+1)*n] : 0.0) +
          r * ((j > 0)   ? u[i+(j-1)*n] : 0.0);
        }
      });
    });

    // Pointer swap
    auto tmp = u;
    u = u_tmp;
    u_tmp = tmp;
  }

  q.wait();
  auto toc = std::chrono::high_resolution_clock::now();

  double *u_host = new double[grid_size];
  q.memcpy(u_host, u, sizeof(double) * grid_size).wait();

  //
  // Check the L2-norm of the computed solution
  // against the *known* solution from the MMS scheme
  //
  double norm = l2norm(n, u_host, nsteps, dt, alpha, dx, length);

  // Stop total timer
  auto stop = std::chrono::high_resolution_clock::now();

  // Print results
  std::cout
    << "Results" << std::endl << std::endl
    << "Error (L2norm): " << norm << std::endl
    << "Solve time (s): " << std::chrono::duration_cast<std::chrono::duration<double>>(toc-tic).count() << std::endl
    << "Total time (s): " << std::chrono::duration_cast<std::chrono::duration<double>>(stop-start).count() << std::endl
    << "Bandwidth (GB/s): " << 1.0E-9*2.0*n*n*nsteps*sizeof(double)/std::chrono::duration_cast<std::chrono::duration<double>>(toc-tic).count() << std::endl
    << LINE << std::endl;

  sycl::free(u, q);
  sycl::free(u_tmp, q);
  delete[] u_host;
}


// True answer given by the manufactured solution
double solution(const double t, const double x, const double y, const double alpha, const double length) {

  return exp(-2.0*alpha*PI*PI*t/(length*length)) * sin(PI*x/length) * sin(PI*y/length);

}


// Computes the L2-norm of the computed grid and the MMS known solution
// The known solution is the same as the boundary function.
double l2norm(const unsigned int n, const double * u, const int nsteps, const double dt, const double alpha, const double dx, const double length) {
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
