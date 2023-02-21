#include <stdio.h>
#include <math.h>
#include <chrono>
#include <cuda.h>
#include "reference.h"
#include "kernels.h"

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  double costheta = 0.3;
  int lmax = LMAX;
  double h_plm[NDLM];
  double r_plm[NDLM]; // reference


  double *d_plm;
  cudaMalloc(&d_plm, NDLM * sizeof(double));

  // warmup up and check results
  bool ok = true;
  for (int l = 0; l <= lmax; l++) { 
    associatedLegendre<<<1, LMAX>>>(costheta,l,d_plm);

    // compute on host
    associatedLegendreFunctionNormalized<double>(costheta,l,r_plm);

    cudaMemcpy(h_plm, d_plm, NDLM * sizeof(double), cudaMemcpyDeviceToHost);  

    for(int i = 0; i <= l; i++)
    {
      if(fabs(h_plm[i] - r_plm[i]) > 1e-6)
      {
        fprintf(stderr, "%d: %lf != %lf\n", i, h_plm[i], r_plm[i]);
        ok = false;
        break;
      }
    }
  }

  printf("%s\n", ok ? "PASS" : "FAIL");

  // performance measurement
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    for (int l = 0; l <= lmax; l++)
      associatedLegendre<<<1, LMAX>>>(costheta,l,d_plm);
  }

  cudaDeviceSynchronize();

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of associatedLegendre kernels %f (us)\n",
         time * 1e-3f / repeat);

  cudaFree(d_plm);

  return 0;
}
