/***************************************************************************
 *
 *            (C) Copyright 2007 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

/*
 * C code for creating the Q data structure for fast convolution-based
 * Hessian multiplication for arbitrary k-space trajectories.
 *
 * Inputs:
 * kx - VECTOR of kx values, same length as ky and kz
 * ky - VECTOR of ky values, same length as kx and kz
 * kz - VECTOR of kz values, same length as kx and ky
 * x  - VECTOR of x values, same length as y and z
 * y  - VECTOR of y values, same length as x and z
 * z  - VECTOR of z values, same length as x and y
 * phi - VECTOR of the Fourier transform of the spatial basis
 *      function, evaluated at [kx, ky, kz].  Same length as kx, ky, and kz.
 *
 * recommended g++ options:
 *  -O3 -lm -ffast-math -funroll-all-loops
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "file.h"
#include "computeQ.cpp"

int main (int argc, char *argv[]) {
  char* inputFileName = argv[1];
  char* outputFileName = argv[2];

  int numX, numK;		/* Number of X and K values */
  float *kx, *ky, *kz;		/* K trajectory (3D vectors) */
  float *x, *y, *z;		/* X coordinates (3D vectors) */
  float *phiR, *phiI;		/* Phi values (complex) */
  float *phiMag;		/* Magnitude of Phi */
  float *Qr, *Qi;		/* Q signal (complex) */

  struct kValues* kVals;

  /* Read in data */
  inputData(inputFileName,
	    &numK, &numX,
	    &kx, &ky, &kz,
	    &x, &y, &z,
	    &phiR, &phiI);

  printf("%d pixels in output; %d samples in trajectory\n", numX, numK);

  /* Create CPU data structures */
  createDataStructsCPU(numK, numX, &phiMag, &Qr, &Qi);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  /* GPU section 1 (precompute PhiMag) */
  /* Mirror several data structures on the device */
  float *phiR_d = sycl::malloc_device<float>(numK, q);
  q.memcpy(phiR_d, phiR, numK * sizeof(float));

  float *phiI_d = sycl::malloc_device<float>(numK, q);
  q.memcpy(phiI_d, phiI, numK * sizeof(float));

  float *phiMag_d = sycl::malloc_device<float>(numK, q);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  computePhiMag_GPU(q, numK, phiR_d, phiI_d, phiMag_d);

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("computePhiMag execution time: %f s\n", time * 1e-9);

  q.memcpy(phiMag, phiMag_d, numK * sizeof(float)).wait();
  sycl::free(phiMag_d, q);
  sycl::free(phiI_d, q);
  sycl::free(phiR_d, q);

  kVals = (struct kValues*)calloc(numK, sizeof (struct kValues));
  for (int k = 0; k < numK; k++) {
    kVals[k].Kx = kx[k];
    kVals[k].Ky = ky[k];
    kVals[k].Kz = kz[k];
    kVals[k].PhiMag = phiMag[k];
  }

  /* GPU section 2 */
  float *x_d = sycl::malloc_device<float>(numX, q);
  q.memcpy(x_d, x, numX * sizeof(float));

  float *y_d = sycl::malloc_device<float>(numX, q);
  q.memcpy(y_d, y, numX * sizeof(float));

  float *z_d = sycl::malloc_device<float>(numX, q);
  q.memcpy(z_d, z, numX * sizeof(float));

  float *Qr_d = sycl::malloc_device<float>(numX, q);
  float *Qi_d = sycl::malloc_device<float>(numX, q);

  q.memset(Qr_d, 0, numX * sizeof(float));
  q.memset(Qi_d, 0, numX * sizeof(float));

  q.wait();
  start = std::chrono::steady_clock::now();

  computeQ_GPU(q, numK, numX, x_d, y_d, z_d, kVals, Qr_d, Qi_d);

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("computeQ time: %f s\n", time * 1e-9);

  q.memcpy(Qr, Qr_d, numX * sizeof(float));
  q.memcpy(Qi, Qi_d, numX * sizeof(float));
  q.wait();
  sycl::free(x_d, q);
  sycl::free(y_d, q);
  sycl::free(z_d, q);
  sycl::free(Qr_d, q);
  sycl::free(Qi_d, q);

  outputData(outputFileName, Qr, Qi, numX);

  free(phiMag);
  free (kx);
  free (ky);
  free (kz);
  free (x);
  free (y);
  free (z);
  free (phiR);
  free (phiI);
  free (kVals);
  free (Qr);
  free (Qi);
  return 0;
}
