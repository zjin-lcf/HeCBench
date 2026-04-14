#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <hip/hip_runtime.h>

#define GPU_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true)
{
   if (code != hipSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

inline int div_roundup(int x , int y) {
  return (x + y - 1)/ y;
}

#endif
