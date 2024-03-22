#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda.h>
#include "block_load.h"
#include "block_store.h"

#define NUM 4

__global__ void reference (const float * __restrict__ A,
                           unsigned char *out, const unsigned int n)
{
  for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < n; idx += gridDim.x * blockDim.x) {
    out[idx] = int(A[idx]);
  }
}

template<int TH, int NUM_BLOCK>
__global__ void kernel (const float * __restrict__ A,
                        unsigned char *out, const unsigned int n)
{
  const int bid = blockIdx.x;
  const int base_idx = (bid * NUM_BLOCK);

  float vals[NUM];
  unsigned char qvals[NUM];

  typedef BlockLoad<float, TH, NUM> LoadFloat;
  typedef BlockStore<unsigned char, TH, NUM> StoreChar;

  __shared__ typename LoadFloat::TempStorage loadf;
  __shared__ typename StoreChar::TempStorage storec;

  for (unsigned int i = base_idx; i < n; i += gridDim.x*NUM_BLOCK)
  {
      unsigned int valid_items = n - i > NUM_BLOCK ? NUM_BLOCK : n - i;

      LoadFloat(loadf).Load(&(A[i]), vals, valid_items);

      #pragma unroll
      for(int j = 0; j < NUM; j++)
          qvals[j] = int(vals[j]);

      StoreChar(storec).Store(&(out[i]), qvals, valid_items);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of rows> <number of columns> <repeat>\n", argv[0]);
    return 1;
  }
  const int nrows = atoi(argv[1]);
  const int ncols = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const unsigned int n = nrows * ncols;
  const size_t A_size = n * sizeof(float);
  const size_t out_size = n * sizeof(unsigned char);

  float *A = (float*) malloc (A_size);
  unsigned char *out = (unsigned char*) malloc (out_size);

  std::mt19937 gen{19937};
 
  std::normal_distribution<float> d{128.0, 127.0};

  for (unsigned int i = 0; i < n; i++) {
    A[i] = d(gen); 
  }

  float *d_A;
  cudaMalloc((void**)&d_A, A_size);
  cudaMemcpy(d_A, A, A_size, cudaMemcpyHostToDevice);

  unsigned char *d_out;
  cudaMalloc((void**)&d_out, out_size);
  
  const int block_size = 256;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  dim3 grid (16 * prop.multiProcessorCount);
  dim3 block (block_size);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    reference<<<grid, block>>>(d_A, d_out, n);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the reference kernel: %f (us)\n", (time * 1e-3f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    kernel<block_size, block_size*NUM><<<grid, block>>>(d_A, d_out, n);
  }

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the blockAccess kernel: %f (us)\n", (time * 1e-3f) / repeat);
  
  cudaMemcpy(out, d_out, out_size, cudaMemcpyDeviceToHost);

  bool error = false;
  for (unsigned int i = 0; i < n; i++) {
    unsigned char t = int(A[i]);
    if (out[i] != t) {
      printf("@%u: %u != %u\n", i, out[i], t);
      error = true;
      break;
    }
  }
  printf("%s\n", error ? "FAIL" : "PASS");
  
  cudaFree(d_A);
  cudaFree(d_out);
  free(A);
  free(out);
  return 0;
}
