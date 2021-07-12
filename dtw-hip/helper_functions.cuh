#pragma once
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
using namespace std;


/** Kernel to fill a matrix with infinity except for index 0 = 0.0
 *  to initialize the DTW cost matrix
 */
__global__ void fill_matrix_inf(float *A, uint width, uint height, float val);


#define cudaErrchk(ans)                                                        \
{                                                                          \
  GPUAssert((ans), __FILE__, __LINE__);                                  \
}
inline void GPUAssert(hipError_t code, const char *file, int line,
    bool abort = true)
{
  if (code != hipSuccess)
  {
    fprintf(stderr, "GPU assert: %s %s %d\n", hipGetErrorString(code),
        file, line);
    if (abort)
    {
      exit(code);
    }
  }
}
