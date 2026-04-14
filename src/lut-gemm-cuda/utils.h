#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <cuda_runtime.h>

#define GPU_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

inline int div_roundup(int x , int y) {
  return (x + y - 1)/ y;
}

#endif
