//
// Program to solve Laplace equation on a regular 3D grid
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include "kernel.h"
#include "reference.h"

// declaration, forward

void reference(int NX, int NY, int NZ, float* h_u1, float* h_u2);

void printHelp(void);

// Main program

int main(int argc, char **argv){

  // 'h_' prefix - CPU (host) memory space

  int    NX, NY, NZ, REPEAT, bx, by, i, j, k, ind, pitch;
  int    verify;
  float  *h_u1, *h_u2, *h_u3, *h_tmp, err;

  // 'd_' prefix - GPU (device) memory space

  float  *d_u1, *d_u2, *d_tmp;

  // check command line inputs

  if(argc != 6) {
    printHelp();
    return 1;
  }

  NX = atoi(argv[1]);
  NY = atoi(argv[2]);
  NZ = atoi(argv[3]);
  REPEAT = atoi(argv[4]);
  verify = atoi(argv[5]);

  if (NX <= 0 || NY <= 0 || NZ <= 0 || REPEAT <= 0) return 1;

  printf("\nGrid dimensions: %d x %d x %d\n", NX, NY, NZ);
  printf("Result verification %s \n", verify ? "enabled" : "disabled");
 
  // allocate memory for arrays

  const size_t grid3D_size = NX * NY * NZ ;
  const size_t grid3D_bytes = grid3D_size * sizeof(float);
  size_t pitch_bytes;

  h_u1 = (float *) malloc (grid3D_bytes);
  h_u2 = (float *) malloc (grid3D_bytes);
  h_u3 = (float *) malloc (grid3D_bytes);
  cudaMallocPitch((void **)&d_u1, &pitch_bytes, sizeof(float)*NX, NY*NZ);
  cudaMallocPitch((void **)&d_u2, &pitch_bytes, sizeof(float)*NX, NY*NZ);
  pitch = pitch_bytes/sizeof(float);

  // initialise u1
    
  for (k=0; k<NZ; k++) {
    for (j=0; j<NY; j++) {
      for (i=0; i<NX; i++) {
        ind = i + j*NX + k*NX*NY;

        if (i==0 || i==NX-1 || j==0 || j==NY-1|| k==0 || k==NZ-1)
          h_u1[ind] = 1.0f;           // Dirichlet b.c.'s
        else
          h_u1[ind] = 0.0f;
      }
    }
  }

  // copy u1 to device
  cudaMemcpy2D(d_u1, pitch_bytes, h_u1, 
               sizeof(float)*NX, sizeof(float)*NX,
               NY*NZ, cudaMemcpyHostToDevice);

  // Set up the execution configuration
  bx = 1 + (NX-1)/BLOCK_X;
  by = 1 + (NY-1)/BLOCK_Y;

  dim3 dimGrid(bx,by);
  dim3 dimBlock(BLOCK_X,BLOCK_Y);

  printf("\n dimGrid  = %d %d %d \n", bx, by, 1);
  printf(" dimBlock = %d %d %d \n", BLOCK_X, BLOCK_Y, 1);

  // Execute GPU kernel
  for (i = 1; i <= REPEAT; ++i) {
    laplace3d<<<dimGrid, dimBlock>>>(NX, NY, NZ, pitch, d_u1, d_u2);
    d_tmp = d_u1; d_u1 = d_u2; d_u2 = d_tmp;   // swap d_u1 and d_u3
  }

  // Read back GPU results
  cudaMemcpy2D(h_u2, sizeof(float)*NX, d_u1, pitch_bytes,
               sizeof(float)*NX, NY*NZ, cudaMemcpyDeviceToHost);

  if (verify) {
    // Reference
    for (i = 1; i <= REPEAT; ++i) {
      reference(NX, NY, NZ, h_u1, h_u3);
      h_tmp = h_u1; h_u1 = h_u3; h_u3 = h_tmp;   // swap h_u1 and h_u3
    }

    // verify (may take long for large grid sizes)
    err = 0.0;
    for (k=0; k<NZ; k++) {
      for (j=0; j<NY; j++) {
        for (i=0; i<NX; i++) {
          ind = i + j*NX + k*NX*NY;
          err += (h_u1[ind]-h_u2[ind])*(h_u1[ind]-h_u2[ind]);
        }
      }
    }
    printf("\n rms error = %f \n",sqrtf(err/ (float)(NX*NY*NZ)));
  }

 // Release GPU and CPU memory
  cudaFree(d_u1);
  cudaFree(d_u2);
  free(h_u1);
  free(h_u2);
  free(h_u3);

  return 0;
}


//Print help screen
void printHelp(void)
{
  printf("Usage:  laplace3d [OPTION]...\n");
  printf("6-point stencil 3D Laplace test \n");
  printf("\n");
  printf("Example: run 100 iterations on a 256x128x128 grid\n");
  printf("./main 256 128 128 100 1\n");

  printf("\n");
  printf("Options:\n");
  printf("Grid width\n");
  printf("Grid height\n");
  printf("Grid depth\n");
  printf("Number of repetitions\n");
  printf("verify the result\n");
}
