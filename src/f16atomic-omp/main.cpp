#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>

#define BLOCK_SIZE 256

using __half = _Float16;

__half __ushort_as_half(unsigned short x)
{
  return *reinterpret_cast<__half*>(&x);
}

void f16AtomicOnGlobalMem(__half* result, int n, int numTeams)
{
   const __half ZERO_FP16 = __ushort_as_half((unsigned short)0x0000U);
   const __half ONE_FP16  = __ushort_as_half((unsigned short)0x3c00U);

  #pragma omp target teams distribute parallel for \
   num_teams(numTeams) num_threads(BLOCK_SIZE)
  for (int tid = 0; tid < n; tid++) {
    int i = tid % BLOCK_SIZE;
    #pragma omp atomic update
    result[2*i+0] += ZERO_FP16;
    #pragma omp atomic update
    result[2*i+1] += ONE_FP16;
  }
}

template <typename T>
void atomicCost (int nelems, int repeat)
{
  int result_size = BLOCK_SIZE * 2;
  size_t result_size_bytes = sizeof(T) * result_size;

  T* result = (T*) malloc (result_size_bytes);

  #pragma omp target data map (alloc: result[0:result_size])
  {

     #pragma omp target teams distribute parallel for
     for (int i = 0; i < result_size; i++) {
       result[i] = 0;
     }

     int numTeams = ((nelems / 2  + BLOCK_SIZE - 1) / BLOCK_SIZE);

     //  warmup
     f16AtomicOnGlobalMem(result, nelems/2, numTeams);
     #pragma omp target update from (result[0:result_size])

     // nelems / 2 / BLOCK_SIZE
     printf("Print the first two elements in HEX: 0x%04x 0x%04x\n", result[0], result[1]);
     printf("Print the first two elements in FLOAT32: %f %f\n\n", (float)result[0], (float)result[1]);

     auto start = std::chrono::steady_clock::now();
     for(int i=0; i<repeat; i++)
     {
       f16AtomicOnGlobalMem(result, nelems/2, numTeams);
     }
     auto end = std::chrono::steady_clock::now();
     auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
     printf("Average execution time of 16-bit floating-point atomic add on global memory: %f (us)\n",
             time * 1e-3f / repeat);
  }
  free(result);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <N> <repeat>\n", argv[0]);
    printf("N: total number of elements (a multiple of 2)\n");
    return 1;
  }
  const int nelems = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  assert(nelems > 0 && (nelems % 2) == 0);

  printf("\nFP16 atomic add\n");
  atomicCost<__half>(nelems, repeat);

  return 0;
}
