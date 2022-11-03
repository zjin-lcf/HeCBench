#include <chrono>
#include <random>
#include <new>
#include <omp.h>
#include "util.h"
#include "kernels.cpp"

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
  float* dx2 = (float*) calloc (ncells, sizeof(float));
  float* dy2 = (float*) calloc (ncells, sizeof(float));
  float* dz2 = (float*) calloc (ncells, sizeof(float));
  float* ra = (float*) calloc (ncells, sizeof(float));
  float* rb = (float*) calloc (ncells, sizeof(float));
  float* da = (float*) calloc (ncells, sizeof(float));
  float* db = (float*) calloc (ncells, sizeof(float));

  build_input_central_cube(ncells, mx, my, mz, a, b, 1.0f, 0.0f, 0.5f, 0.25f, 0.05f);

#pragma omp target data map (tofrom: a[0:ncells], b[0:ncells]) \
                        map (to: dx2[0:ncells], dy2[0:ncells], dz2[0:ncells], \
                                 ra[0:ncells], rb[0:ncells], da[0:ncells], db[0:ncells])
{
  // set constants
  float diffcon_a = Da / (dx * dx);
  float diffcon_b = Db / (dx * dx);

  // keep track of time
  auto start = std::chrono::system_clock::now();

  for(unsigned int t=0; t<timesteps; t++) {

    // calculate laplacian for A
    if(zeroflux) {
      // x2 derivative
      derivative_x2_zeroflux(a, dx2, mx, my, mz, pencils);

      // y2 derivative
      derivative_y2_zeroflux(a, dy2, mx, my, mz, pencils);

      // z2 derivative
      derivative_z2_zeroflux(a, dz2, mx, my, mz, pencils);
    } else {
      // x2 derivative
      derivative_x2_pbc(a, dx2, mx, my, mz, pencils);

      // y2 derivative
      derivative_y2_pbc(a, dy2, mx, my, mz, pencils);

      // z2 derivative
      derivative_z2_pbc(a, dz2, mx, my, mz, pencils);
    }

    // sum all three derivative components
    construct_laplacian(da, dx2, dy2, dz2, ncells, diffcon_a);

    // calculate laplacian for B
    if(zeroflux) {
      // x2 derivative
      derivative_x2_zeroflux(b, dx2, mx, my, mz, pencils);

      // y2 derivative
      derivative_y2_zeroflux(b, dy2, mx, my, mz, pencils);

      // z2 derivative
      derivative_z2_zeroflux(b, dz2, mx, my, mz, pencils);
    } else {
      // x2 derivative
      derivative_x2_pbc(b, dx2, mx, my, mz, pencils);

      // y2 derivative
      derivative_y2_pbc(b, dy2, mx, my, mz, pencils);

      // z2 derivative
      derivative_z2_pbc(b, dz2, mx, my, mz, pencils);
    }

    // sum all three derivative components
    construct_laplacian(db, dx2, dy2, dz2, ncells, diffcon_b);

    // calculate reaction
    reaction_gray_scott(a, b, ra, rb, ncells, c1, c2);

    // update
    update(a, b, da, db, ra, rb, ncells, dt);
  }

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  printf("timesteps: %d\n", timesteps);
  printf("Total kernel execution time:     %12.3f s\n\n", elapsed_seconds.count());
}
  // output lowest and highest values
  stats(a, b, ncells);

  delete [] a;
  delete [] b;
  return 0;
}
