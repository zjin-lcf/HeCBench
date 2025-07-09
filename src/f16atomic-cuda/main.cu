#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

static void CheckError( cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
  }
}
#define CHECK_ERROR( err ) (CheckError( err, __FILE__, __LINE__ ))

#define BLOCK_SIZE 256

#define ZERO_FP16 __ushort_as_half((unsigned short)0x0000U)
#define ONE_FP16  __ushort_as_half((unsigned short)0x3c00U)
#define ZERO_BF16 __ushort_as_bfloat16((unsigned short)0x0000U)
#define ONE_BF16  __ushort_as_bfloat16((unsigned short)0x3f80U)

__global__
void f16AtomicOnGlobalMem(__half* result, int n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  __half2 *result_v = reinterpret_cast<__half2*>(result);
  __half2 val {ZERO_FP16, ONE_FP16};
  atomicAdd(&result_v[tid % BLOCK_SIZE], val);
}

__global__
void f16AtomicOnGlobalMem(__nv_bfloat16* result, int n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  __nv_bfloat162 *result_v = reinterpret_cast<__nv_bfloat162*>(result);
  __nv_bfloat162 val {ZERO_BF16, ONE_BF16};
  atomicAdd(&result_v[tid % BLOCK_SIZE], val);
}

template <typename T>
void atomicCost (int nelems, int repeat)
{
  size_t result_size = sizeof(T) * BLOCK_SIZE * 2;

  T* result = (T*) malloc (result_size);

  T *d_result;
  CHECK_ERROR( cudaMalloc((void **)&d_result, result_size) );
  CHECK_ERROR( cudaMemset(d_result, 0, result_size) );

  dim3 block (BLOCK_SIZE);
  dim3 grid ((nelems / 2  + BLOCK_SIZE - 1) / BLOCK_SIZE);

  //  warmup
  f16AtomicOnGlobalMem<<<grid, block>>>(d_result, nelems/2);
  CHECK_ERROR( cudaMemcpy(result, d_result, result_size, cudaMemcpyDeviceToHost) );
  // nelems / 2 / BLOCK_SIZE
  printf("Print the first two elements in HEX: 0x%04x 0x%04x\n", result[0], result[1]);
  printf("Print the first two elements in FLOAT32: %f %f\n\n", (float)result[0], (float)result[1]);


  CHECK_ERROR( cudaDeviceSynchronize() );
  auto start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    f16AtomicOnGlobalMem<<<grid, block>>>(d_result, nelems/2);
  }
  CHECK_ERROR( cudaDeviceSynchronize() );
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of 16-bit floating-point atomic add on global memory: %f (us)\n",
          time * 1e-3f / repeat);
  free(result);
  CHECK_ERROR(cudaFree(d_result));
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

  printf("\nBF16 atomic add\n");
  atomicCost<__nv_bfloat16>(nelems, repeat);

  return 0;
}
