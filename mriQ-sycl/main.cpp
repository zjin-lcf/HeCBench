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
#include "common.h"
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
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif

  queue q(dev_sel);

  /* GPU section 1 (precompute PhiMag) */
  auto start = std::chrono::steady_clock::now();
  {
    /* Mirror several data structures on the device */
    buffer<float, 1> phiR_d (phiR, numK);
    buffer<float, 1> phiI_d (phiI, numK);
    buffer<float, 1> phiMag_d (phiMag, numK);

    computePhiMag_GPU(q, numK, phiR_d, phiI_d, phiMag_d);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("computePhiMag execution time: %f s\n", time * 1e-9);

  kVals = (struct kValues*)calloc(numK, sizeof (struct kValues));
  for (int k = 0; k < numK; k++) {
    kVals[k].Kx = kx[k];
    kVals[k].Ky = ky[k];
    kVals[k].Kz = kz[k];
    kVals[k].PhiMag = phiMag[k];
  }

  /* GPU section 2 */
  start = std::chrono::steady_clock::now();
  {
    buffer<float, 1> x_d (x, numX);
    buffer<float, 1> y_d (y, numX);
    buffer<float, 1> z_d (z, numX);
    buffer<float, 1> Qr_d (Qr, numX);
    buffer<float, 1> Qi_d (Qi, numX);

    q.submit([&] (handler &cgh) {
      auto acc = Qr_d.get_access<sycl_discard_write>(cgh);
      cgh.fill(acc, 0.f);
    });

    q.submit([&] (handler &cgh) {
      auto acc = Qi_d.get_access<sycl_discard_write>(cgh);
      cgh.fill(acc, 0.f);
    });

    computeQ_GPU(q, numK, numX, x_d, y_d, z_d, kVals, Qr_d, Qi_d);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("computeQ execution time: %f s\n", time * 1e-9);

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
