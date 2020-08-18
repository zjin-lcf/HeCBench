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
**          Ported to OpenCL by Tom Deakin, Jan 2020
**
*/

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>

// Key constants used in this program
#define PI sycl::acos(-1.0) // Pi
#define LINE "--------------------" // A line for fancy output

// Function definitions
void initial_value(const unsigned int n, const double dx, const double length, double * u,
                   sycl::nd_item<3> item_ct1);
void zero(const unsigned int n, double * u, sycl::nd_item<3> item_ct1);
void solve(const unsigned int n, const double alpha, const double dx, const double dt, const double r, const double r2,
		double * __restrict__ u, double * __restrict__ u_tmp,
		sycl::nd_item<3> item_ct1);
double solution(const double t, const double x, const double y, const double alpha, const double length);
double l2norm(const int n, const double * u, const int nsteps, const double dt, const double alpha, const double dx, const double length);


// Main function
int main(int argc, char *argv[]) try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  // Start the total program runtime timer
  auto start = std::chrono::high_resolution_clock::now();

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

  dpct::device_info prop;
  dpct::dev_mgr::instance().get_device(0).get_device_info(prop);
  char *device_name = prop.get_name();

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
    << " HIP device: " << device_name << std::endl
    << LINE << std::endl;




  // Stability check
  std::cout << "Stability" << std::endl << std::endl;
  std::cout << " r value: " << r << std::endl;
  if (r > 0.5)
    std::cout << " Warning: unstable" << std::endl;
  std::cout << LINE << std::endl;


  // Allocate two nxn grids
  double *u;
  double *u_tmp;
  dpct::dpct_malloc((void **)&u, sizeof(double) * n * n);
  dpct::dpct_malloc((void **)&u_tmp, sizeof(double) * n * n);

  // Set the initial value of the grid under the MMS scheme
  const int block_size = 256;
  int n_ceil = (n*n+block_size-1) / block_size;
  sycl::range<3> grid(n_ceil, 1, 1);
  sycl::range<3> block(block_size, 1, 1);
  {
    dpct::buffer_t u_buf_ct3 = dpct::get_buffer(u);
    q_ct1.submit([&](sycl::handler &cgh) {
      auto u_acc_ct3 =
          u_buf_ct3.get_access<sycl::access::mode::read_write>(cgh);

      auto dpct_global_range = sycl::range<3>(grid) * sycl::range<3>(block);
      auto dpct_local_range = sycl::range<3>(block);

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                             dpct_global_range.get(0)),
              sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1),
                             dpct_local_range.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            initial_value(n, dx, length, (double *)(&u_acc_ct3[0]), item_ct1);
          });
    });
  }
  {
    dpct::buffer_t u_tmp_buf_ct1 = dpct::get_buffer(u_tmp);
    q_ct1.submit([&](sycl::handler &cgh) {
      auto u_tmp_acc_ct1 =
          u_tmp_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);

      auto dpct_global_range = sycl::range<3>(grid) * sycl::range<3>(block);
      auto dpct_local_range = sycl::range<3>(block);

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                             dpct_global_range.get(0)),
              sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1),
                             dpct_local_range.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            zero(n, (double *)(&u_tmp_acc_ct1[0]), item_ct1);
          });
    });
  }

  // Ensure everything is initalised on the device
  /*
  DPCT1003:4: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  int err = (dev_ct1.queues_wait_and_throw(), 0);
  /*
  DPCT1000:1: Error handling if-stmt was detected but could not be rewritten.
  */
  if (err != 0) {
    /*
    DPCT1001:0: The statement could not be removed.
    */
    std::cerr << "CUDA error after initalisation" << std::endl;
    exit(EXIT_FAILURE);
  }

  //
  // Run through timesteps under the explicit scheme
  //
  // Finite difference constant multiplier
  const double r2 = 1.0 - 4.0*r;


  // Start the solve timer
  auto tic = std::chrono::high_resolution_clock::now();
  for (int t = 0; t < nsteps; ++t) {

    // Call the solve kernel
    // Computes u_tmp at the next timestep
    // given the value of u at the current timestep
    {
      dpct::buffer_t u_buf_ct6 = dpct::get_buffer(u);
      dpct::buffer_t u_tmp_buf_ct7 = dpct::get_buffer(u_tmp);
      q_ct1.submit([&](sycl::handler &cgh) {
        auto u_acc_ct6 =
            u_buf_ct6.get_access<sycl::access::mode::read_write>(cgh);
        auto u_tmp_acc_ct7 =
            u_tmp_buf_ct7.get_access<sycl::access::mode::read_write>(cgh);

        auto dpct_global_range = sycl::range<3>(grid) * sycl::range<3>(block);
        auto dpct_local_range = sycl::range<3>(block);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                             dpct_global_range.get(1),
                                             dpct_global_range.get(0)),
                              sycl::range<3>(dpct_local_range.get(2),
                                             dpct_local_range.get(1),
                                             dpct_local_range.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
              solve(n, alpha, dx, dt, r, r2, (double *)(&u_acc_ct6[0]),
                    (double *)(&u_tmp_acc_ct7[0]), item_ct1);
            });
      });
    }

    // Pointer swap
    auto tmp = u;
    u = u_tmp;
    u_tmp = tmp;
  }

  // Stop solve timer
  dev_ct1.queues_wait_and_throw();
  auto toc = std::chrono::high_resolution_clock::now();

  // Get access to u on the host
  double *u_host = new double[n*n];
  /*
  DPCT1003:5: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  err = (dpct::dpct_memcpy(u_host, u, sizeof(double) * n * n,
                           dpct::device_to_host),
         0);
  /*
  DPCT1000:3: Error handling if-stmt was detected but could not be rewritten.
  */
  if (err != 0) {
    /*
    DPCT1001:2: The statement could not be removed.
    */
    std::cerr << "CUDA error on copying back data" << std::endl;
    exit(EXIT_FAILURE);
  }

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

  delete[] u_host;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// Sets the mesh to an initial value, determined by the MMS scheme
void initial_value(const unsigned int n, const double dx, const double length, double * u,
                   sycl::nd_item<3> item_ct1) {

  int idx = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);
  if (idx < n*n) {
    int i = idx % n;
    int j = idx / n;
    double y = dx * (j+1); // Physical y position
    double x = dx * (i+1); // Physical x position
    u[i + j * n] = sycl::sin(PI * x / length) * sycl::sin(PI * y / length);
  }
}


// Zero the array u
void zero(const unsigned int n, double * u, sycl::nd_item<3> item_ct1) {

  int idx = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);
  if (idx < n*n) u[idx] = 0.0;
}


// Compute the next timestep, given the current timestep
// Loop over the nxn grid
void solve(const unsigned int n, const double alpha, const double dx, const double dt, 
		const double r, const double r2,
		double * __restrict__ u, double * __restrict__ u_tmp,
		sycl::nd_item<3> item_ct1) {

  int idx = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);
  if (idx < n * n) {
    int i = idx % n;
    int j = idx / n;
    // Boundaries are zero because the MMS solution is zero there.
    u_tmp[i+j*n] =  r2 * u[i+j*n] +
    r * ((i < n-1) ? u[i+1+j*n] : 0.0) +
    r * ((i > 0)   ? u[i-1+j*n] : 0.0) +
    r * ((j < n-1) ? u[i+(j+1)*n] : 0.0) +
    r * ((j > 0)   ? u[i+(j-1)*n] : 0.0);
  }
}


// True answer given by the manufactured solution
double solution(const double t, const double x, const double y, const double alpha, const double length) {

  return exp(-2.0 * alpha * PI * PI * t / (length * length)) *
         sin(PI * x / length) * sin(PI * y / length);
}


// Computes the L2-norm of the computed grid and the MMS known solution
// The known solution is the same as the boundary function.
double l2norm(const int n, const double * u, const int nsteps, const double dt, const double alpha, const double dx, const double length) {

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

