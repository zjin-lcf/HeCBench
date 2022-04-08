#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include "reference.h"

template <class T>
void testcase(const int repeat)
{
  const int len = 1 << 10;
  unsigned int numThreads = 256;
  unsigned int numData = 7;
  unsigned int memSize = sizeof(T) * numData;
  const T data[] = {0, 0, (T)-256, 256, 255, 0, 255};
  T gpuData[7];

  #pragma omp target data map(from: gpuData[0:7])
  {
    for (int n = 0; n < repeat; n++) {
      memcpy(gpuData, data, memSize);
      #pragma omp target update to (gpuData[0:7])

      #pragma omp target teams distribute parallel for thread_limit(numThreads)
      for (int i = 0; i < len; ++i)
      {
         #pragma omp atomic update  
          gpuData[0] += (T)10;
         #pragma omp atomic update  
          gpuData[1] -= (T)10;
         //#pragma omp atomic compare  
         // gpuData[2] = (gpuData[2] < number[i]) ? number[i] : gpuData[2];
         //#pragma omp atomic compare  
         // gpuData[3] = (gpuData[3] > number[i]) ? number[i] : gpuData[3];
         #pragma omp atomic update  
          gpuData[4] &= (T)(2*i+7);
         #pragma omp atomic update  
          gpuData[5] |= (T)(1 << i);
         #pragma omp atomic update  
          gpuData[6] ^= (T)i;
      }

      #pragma omp target teams distribute parallel for thread_limit(256) reduction(max: gpuData[2])
      for (int i = 0; i < len; ++i)
         gpuData[2] = max(gpuData[2], (T)i);

      #pragma omp target teams distribute parallel for thread_limit(256) reduction(min: gpuData[3])
      for (int i = 0; i < len; ++i)
         gpuData[3] = min(gpuData[3], (T)i);
    }
  }
  computeGold<T>(gpuData, len);
}

int main(int argc, char **argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  testcase<int>(repeat);
  testcase<unsigned int>(repeat);
  return 0;
}
