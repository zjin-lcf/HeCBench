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
#include <omp.h>
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

  /* GPU section 1 (precompute PhiMag) */
  int phiMagBlocks = numK / KERNEL_PHI_MAG_THREADS_PER_BLOCK;
  if (numK % KERNEL_PHI_MAG_THREADS_PER_BLOCK)
    phiMagBlocks++;

  #pragma omp target data map(to: phiR[0:numK], phiI[0:numK]) \
                          map(from: phiMag[0:numK])
  {
    auto start = std::chrono::steady_clock::now();

    #pragma omp target teams distribute parallel for \
      num_teams(phiMagBlocks) thread_limit(KERNEL_PHI_MAG_THREADS_PER_BLOCK)
    for (int indexK = 0; indexK < numK; indexK++) {
      float real = phiR[indexK];
      float imag = phiI[indexK];
      phiMag[indexK] = real*real + imag*imag;
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("computePhiMag time: %f s\n", time * 1e-9f);
  }

  kVals = (struct kValues*)calloc(numK, sizeof (struct kValues));
  for (int k = 0; k < numK; k++) {
    kVals[k].Kx = kx[k];
    kVals[k].Ky = ky[k];
    kVals[k].Kz = kz[k];
    kVals[k].PhiMag = phiMag[k];
  }

  /* GPU section 2 */
  kValues ck [KERNEL_Q_K_ELEMS_PER_GRID];

  #pragma omp target data map(to: x[0:numX], y[0:numX], z[0:numX]) \
                          map(from: Qr[0:numX], Qi[0:numX]) \
                          map(alloc: ck[0:KERNEL_Q_K_ELEMS_PER_GRID])
  {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < numX; i++) {
      Qr[i] = 0.f;
      Qi[i] = 0.f;
    }

    auto start = std::chrono::steady_clock::now();

    computeQ_GPU(numK, numX, x, y, z, kVals, ck, Qr, Qi);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("computeQ time: %f s\n", time * 1e-9f);
  }

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
