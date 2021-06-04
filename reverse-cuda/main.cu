#include <stdio.h>
#include <assert.h>
#include <cuda.h>

__global__ void reverse (int* d, const int len)
{
  __shared__ int s[256];
  int t = threadIdx.x;
  int tr = len-t-1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}

int main() {
  const int len = 256;
  const int iteration = 1 << 20;
  int d[len];
  for (int i = 0; i < len; i++) d[i] = i;

  int *dd;
  cudaMalloc((void**)&dd, sizeof(int)*len);
  cudaMemcpy(dd, d, sizeof(int)*len, cudaMemcpyHostToDevice);
  for (int i = 0; i <= iteration; i++)
    reverse<<<1, 256>>> (dd, len);
  cudaMemcpy(d, dd, sizeof(int)*len, cudaMemcpyDeviceToHost);
  cudaFree(dd);

  for (int i = 0; i < len; i++) assert(d[i] == len-i-1);
  printf("PASS\n");

  return 0;
}
