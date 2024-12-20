#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

static void CheckError( hipError_t err, const char *file, int line ) {
  if (err != hipSuccess) {
    printf( "%s in %s at line %d\n", hipGetErrorString( err ), file, line );
  }
}
#define CHECK_ERROR( err ) (CheckError( err, __FILE__, __LINE__ ))

#define BLOCK_SIZE 256

#define ONE_FP16  __ushort_as_half((unsigned short)0x3C00U)
#define ZERO_FP16 __ushort_as_half((unsigned short)0x0000U)
#define ONE_BF16  __ushort_as_bfloat16((unsigned short)0x3F80U)
#define ZERO_BF16 __ushort_as_bfloat16((unsigned short)0x0000U)

template <typename T>
__global__
void f16AtomicOnGlobalMem(T* result, int repeat)
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < repeat; i++)
    atomicAdd(&result[tid], tid % 2);
}

template <>
__global__
void f16AtomicOnGlobalMem<__half>(__half* result, int repeat)
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  __half2 *result_v = reinterpret_cast<__half2*>(result);
  __half2 val {ZERO_FP16, ONE_FP16};
  for (int i = 0; i < repeat; i++)
    unsafeAtomicAdd(&result_v[tid], val);
}

template <>
__global__
void f16AtomicOnGlobalMem<__hip_bfloat16>(__hip_bfloat16* result, int repeat)
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  __hip_bfloat162 *result_v = reinterpret_cast<__hip_bfloat162*>(result);
  __hip_bfloat162 val {ZERO_BF16, ONE_BF16};
  for (int i = 0; i < repeat; i++)
    unsafeAtomicAdd(&result_v[tid], val);
}

template <typename T>
void atomicCost (int nelems, int repeat)
{
  size_t result_size = sizeof(T) * nelems;

  T* result = (T*) malloc (result_size);

  T *d_result;
  CHECK_ERROR( hipMalloc((void **)&d_result, result_size) );
  CHECK_ERROR( hipMemset(d_result, 0, result_size) );

  dim3 block (BLOCK_SIZE);
  dim3 grid (nelems / 2 / BLOCK_SIZE);

  const int atomics_count = 16;

  CHECK_ERROR( hipDeviceSynchronize() );
  auto start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    f16AtomicOnGlobalMem<T><<<grid, block>>>(d_result, atomics_count);
  }
  CHECK_ERROR( hipDeviceSynchronize() );
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of 16-bit floating-point atomic add on global memory: %f (us)\n",
          time * 1e-3f / repeat);
  CHECK_ERROR( hipMemcpy(result, d_result, result_size, hipMemcpyDeviceToHost) );

  bool error = false;
  for (int i = 0; i < nelems; i=i+2) {
    if (float(result[i]) != 0 || float(result[i+1]) != atomics_count * repeat) {
      printf("Error @%d: %f %f\n", i, float(result[i]), float(result[i+1]));
      error = true;
      break;
    }
  }
  printf("%s\n", error ? "FAIL" : "PASS");

  free(result);
  hipFree(d_result);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <N> <repeat>\n", argv[0]);
    printf("N: total number of elements (a multiple of %d)\n", 2*BLOCK_SIZE);
    return 1;
  }
  const int nelems = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  assert(nelems > 0 && (nelems % (2*BLOCK_SIZE)) == 0);

  printf("\nFP16 atomic add\n");
  atomicCost<__half>(nelems, repeat);

  // supported since ROCm 6.3.0
  printf("\nBF16 atomic add\n");
  atomicCost<__hip_bfloat16>(nelems, repeat);

  return 0;
}
