#include <chrono>
#include <random>
#include <new>
#include <hip/hip_runtime.h>
#include "util.h"
#include "kernels.cu"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <timesteps>\n", argv[0]);
    return 1;
  }
  unsigned int timesteps = atoi(argv[1]);

  unsigned int mx = 128;
  unsigned int my = 128;
  unsigned int mz = 128;
  unsigned int ncells = mx * my * mz;
  unsigned int pencils = 2;
  bool zeroflux = true;

  // reaction settings of kinetic system
  float Da = 0.16;            // diffusion constant of A
  float Db = 0.08;            // diffusion constant of B
  float dt = 0.25;            // temporal discretization
  float dx = 0.5;             // spatial discretization

  // generalized kinetic parameters
  float c1 = 0.0392;
  float c2 = 0.0649;

  printf("Starting time-integration\n");
  // build initial concentrations
  printf("Constructing initial concentrations...\n");
  // concentration of components A and B
  float* a = new float[ncells];
  float* b = new float[ncells];

  build_input_central_cube(ncells, mx, my, mz, a, b, 1.0f, 0.0f, 0.5f, 0.25f, 0.05f);

  // device variables
  float *d_a, *d_b, *d_dx2, *d_dy2, *d_dz2, *d_ra, *d_rb, *d_da, *d_db;

  const int bytes = ncells * sizeof(float);
  hipMalloc((void**)&d_a, bytes);
  hipMalloc((void**)&d_b, bytes);
  hipMalloc((void**)&d_dx2, bytes);
  hipMalloc((void**)&d_dy2, bytes);
  hipMalloc((void**)&d_dz2, bytes);
  hipMalloc((void**)&d_ra, bytes);
  hipMalloc((void**)&d_rb, bytes);
  hipMalloc((void**)&d_da, bytes);
  hipMalloc((void**)&d_db, bytes);

  // copy data to device
  hipMemcpy(d_a, a, bytes, hipMemcpyHostToDevice);
  hipMemcpy(d_b, b, bytes, hipMemcpyHostToDevice);
  hipMemset(d_dx2, 0, bytes);
  hipMemset(d_dy2, 0, bytes);
  hipMemset(d_dz2, 0, bytes);
  hipMemset(d_ra, 0, bytes);
  hipMemset(d_rb, 0, bytes);
  hipMemset(d_da, 0, bytes);
  hipMemset(d_db, 0, bytes);

  // set constants
  float diffcon_a = Da / (dx * dx);
  float diffcon_b = Db / (dx * dx);

  dim3 gridx(my / pencils, mz, 1);
  dim3 blockx(mx, pencils, 1);
  dim3 gridy(mx / pencils, mz, 1);
  dim3 blocky(pencils, my, 1);
  dim3 gridz(mx / pencils, my, 1);
  dim3 blockz(pencils, mz, 1);
  unsigned int block = mx;
  unsigned int grid = (ncells + mx - 1) / mx;

  unsigned shared_mem_size = 0;
  if(zeroflux) {
    shared_mem_size = pencils * mx * sizeof(float);
  } else {
    shared_mem_size = pencils * (mx + 2) * sizeof(float);
  }

  // keep track of time
  hipDeviceSynchronize();
  auto start = std::chrono::system_clock::now();

  for(unsigned int t=0; t<timesteps; t++) {

    // calculate laplacian for A
    if(zeroflux) {
      // x2 derivative
      hipLaunchKernelGGL(derivative_x2_zeroflux, gridx, blockx, shared_mem_size, 0, d_a, d_dx2, mx, my);

      // y2 derivative
      hipLaunchKernelGGL(derivative_y2_zeroflux, gridy, blocky, shared_mem_size, 0, d_a, d_dy2, mx, my, pencils);

      // z2 derivative
      hipLaunchKernelGGL(derivative_z2_zeroflux, gridz, blockz, shared_mem_size, 0, d_a, d_dz2, mx, my, mz, pencils);
    } else {
      // x2 derivative
      hipLaunchKernelGGL(derivative_x2_pbc, gridx, blockx, shared_mem_size, 0, d_a, d_dx2, mx, my, pencils);

      // y2 derivative
      hipLaunchKernelGGL(derivative_y2_pbc, gridy, blocky, shared_mem_size, 0, d_a, d_dy2, mx, my, pencils);

      // z2 derivative
      hipLaunchKernelGGL(derivative_z2_pbc, gridz, blockz, shared_mem_size, 0, d_a, d_dz2, mx, my, mz, pencils);
    }

    // sum all three derivative components
    hipLaunchKernelGGL(construct_laplacian, grid, block, 0, 0, d_da, d_dx2, d_dy2, d_dz2, ncells, diffcon_a);

    // calculate laplacian for B
    if(zeroflux) {
      // x2 derivative
      hipLaunchKernelGGL(derivative_x2_zeroflux, gridx, blockx, shared_mem_size, 0, d_b, d_dx2, mx, my);

      // y2 derivative
      hipLaunchKernelGGL(derivative_y2_zeroflux, gridy, blocky, shared_mem_size, 0, d_b, d_dy2, mx, my, pencils);

      // z2 derivative
      hipLaunchKernelGGL(derivative_z2_zeroflux, gridz, blockz, shared_mem_size, 0, d_b, d_dz2, mx, my, mz, pencils);
    } else {
      // x2 derivative
      hipLaunchKernelGGL(derivative_x2_pbc, gridx, blockx, shared_mem_size, 0, d_b, d_dx2, mx, my, pencils);

      // y2 derivative
      hipLaunchKernelGGL(derivative_y2_pbc, gridy, blocky, shared_mem_size, 0, d_b, d_dy2, mx, my, pencils);

      // z2 derivative
      hipLaunchKernelGGL(derivative_z2_pbc, gridz, blockz, shared_mem_size, 0, d_b, d_dz2, mx, my, mz, pencils);
    }

    // sum all three derivative components
    hipLaunchKernelGGL(construct_laplacian, grid, block, 0, 0, d_db, d_dx2, d_dy2, d_dz2, ncells, diffcon_b);

    // calculate reaction
    hipLaunchKernelGGL(reaction_gray_scott, grid, block, 0, 0, d_a, d_b, d_ra, d_rb, ncells, c1, c2);

    // update
    hipLaunchKernelGGL(update, grid, block, 0, 0, d_a, d_b, d_da, d_db, d_ra, d_rb, ncells, dt);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  printf("timesteps: %d\n", timesteps);
  printf("Total kernel execution time:     %12.3f s\n\n", elapsed_seconds.count());

  // copy results back
  hipMemcpy(a, d_a, bytes, hipMemcpyDeviceToHost);
  hipMemcpy(b, d_b, bytes, hipMemcpyDeviceToHost);

  // output lowest and highest values
  stats(a, b, ncells);

  hipFree(d_a);
  hipFree(d_b);
  hipFree(d_ra);
  hipFree(d_rb);
  hipFree(d_da);
  hipFree(d_db);
  hipFree(d_dx2);
  hipFree(d_dy2);
  hipFree(d_dz2);

  delete [] a;
  delete [] b;
  return 0;
}
