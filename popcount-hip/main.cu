#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>

#define m1  0x5555555555555555
#define m2  0x3333333333333333 
#define m4  0x0f0f0f0f0f0f0f0f 
#define h01 0x0101010101010101

#define BLOCK_SIZE 256

// reference implementation
int popcount_ref(unsigned long x)
{
  int count;
  for (count=0; x; count++)
    x &= x - 1;
  return count;
}

// HIP kernels
__global__ void pc1 (const unsigned long* __restrict data, int* __restrict r, const int length)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i >= length) return;
  unsigned long x = data[i];
  x -= (x >> 1) & m1;             //put count of each 2 bits into those 2 bits
  x = (x & m2) + ((x >> 2) & m2); //put count of each 4 bits into those 4 bits 
  x = (x + (x >> 4)) & m4;        //put count of each 8 bits into those 8 bits 
  x += x >>  8;  //put count of each 16 bits into their lowest 8 bits
  x += x >> 16;  //put count of each 32 bits into their lowest 8 bits
  x += x >> 32;  //put count of each 64 bits into their lowest 8 bits
  r[i] = x & 0x7f;
}

__global__ void pc2 (const unsigned long* __restrict data, int* __restrict r, const int length)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i >= length) return;
  unsigned long x = data[i];
  x -= (x >> 1) & m1;             //put count of each 2 bits into those 2 bits
  x = (x & m2) + ((x >> 2) & m2); //put count of each 4 bits into those 4 bits 
  x = (x + (x >> 4)) & m4;        //put count of each 8 bits into those 8 bits 
  r[i] = (x * h01) >> 56;  //returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ... 
}

__global__ void pc3 (const unsigned long* __restrict data, int* __restrict r, const int length)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i >= length) return;
  char count;
  unsigned long x = data[i];
  for (count=0; x; count++) x &= x - 1;
  r[i] = count;
}

__global__ void pc4 (const unsigned long* __restrict data, int* __restrict r, const int length)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i >= length) return;
  unsigned long x = data[i];
  char cnt = 0;
  for (char i = 0; i < 64; i++)
  {
    cnt = cnt + (x & 0x1);
    x = x >> 1;
  }
  r[i] = cnt;
}

__global__ void pc5 (const unsigned long* __restrict data, int* __restrict r, const int length)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i >= length) return;
  unsigned long x = data[i];
  const unsigned char a[256] = { 0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8};
  const unsigned char b[256] = { 0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8};
  const unsigned char c[256] = { 0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8};
  const unsigned char d[256] = { 0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8};

  unsigned char i1 = a[(x & 0xFF)];
  unsigned char i2 = a[(x >> 8) & 0xFF];
  unsigned char i3 = b[(x >> 16) & 0xFF];
  unsigned char i4 = b[(x >> 24) & 0xFF];
  unsigned char i5 = c[(x >> 32) & 0xFF];
  unsigned char i6 = c[(x >> 40) & 0xFF];
  unsigned char i7 = d[(x >> 48) & 0xFF];
  unsigned char i8 = d[(x >> 56) & 0xFF];
  r[i] = (i1+i2)+(i3+i4)+(i5+i6)+(i7+i8);
}

__global__ void pc6 (const unsigned long* __restrict data, int* __restrict r, const int length)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i >= length) return;
  r[i] = __popcll(data[i]);
}

void checkResults(const unsigned long *d, const int *r, const int length)
{
  int error = 0;
  for (int i=0;i<length;i++)
    if (popcount_ref(d[i]) != r[i]) {
      error = 1;
      break;
    }

  if (error)
    printf("Fail\n");
  else
    printf("Success\n");
}

int main(int argc, char* argv[])
{
  const int length = atoi(argv[1]);
  unsigned long *data = NULL;
  int* __restrict result = NULL;
  posix_memalign((void**)&data, 1024, length*sizeof(unsigned long));
  posix_memalign((void**)&result, 1024, length*sizeof(int));

  // initialize input
  srand(2);
  for (int i = 0; i < length; i++) {
    unsigned long t = (unsigned long)rand() << 32;
    data[i] = t | rand();
  }

  // run each popcount implementation 100 times

  unsigned long* d_data;
  hipMalloc((void**)&d_data, sizeof(unsigned long)*length);
  hipMemcpy(d_data, data, sizeof(unsigned long)*length, hipMemcpyHostToDevice);

  int* d_result;
  hipMalloc((void**)&d_result, sizeof(int)*length);

  dim3 grids ((length+BLOCK_SIZE-1)/BLOCK_SIZE);
  dim3 threads (BLOCK_SIZE);

  for (int n = 0; n < 100; n++) {
    hipLaunchKernelGGL(pc1, dim3(grids), dim3(threads), 0, 0, d_data, d_result, length);
  }
  hipMemcpy(result, d_result, sizeof(int)*length, hipMemcpyDeviceToHost);
  checkResults(data, result, length);
  //========================================================================================

  for (int n = 0; n < 100; n++) {
    hipLaunchKernelGGL(pc2, dim3(grids), dim3(threads), 0, 0, d_data, d_result, length);
  }
  hipMemcpy(result, d_result, sizeof(int)*length, hipMemcpyDeviceToHost);
  checkResults(data, result, length);
  //========================================================================================

  for (int n = 0; n < 100; n++) {
    hipLaunchKernelGGL(pc3, dim3(grids), dim3(threads), 0, 0, d_data, d_result, length);
  }
  hipMemcpy(result, d_result, sizeof(int)*length, hipMemcpyDeviceToHost);
  checkResults(data, result, length);
  //========================================================================================

  for (int n = 0; n < 100; n++) {
    hipLaunchKernelGGL(pc4, dim3(grids), dim3(threads), 0, 0, d_data, d_result, length);
  }
  hipMemcpy(result, d_result, sizeof(int)*length, hipMemcpyDeviceToHost);
  checkResults(data, result, length);
  //========================================================================================

  for (int n = 0; n < 100; n++) {
    hipLaunchKernelGGL(pc5, dim3(grids), dim3(threads), 0, 0, d_data, d_result, length);
  }
  hipMemcpy(result, d_result, sizeof(int)*length, hipMemcpyDeviceToHost);
  checkResults(data, result, length);
  //========================================================================================

  for (int n = 0; n < 100; n++) {
    hipLaunchKernelGGL(pc6, dim3(grids), dim3(threads), 0, 0, d_data, d_result, length);
  }
  hipMemcpy(result, d_result, sizeof(int)*length, hipMemcpyDeviceToHost);
  checkResults(data, result, length);
  //========================================================================================

  hipFree(d_data);
  hipFree(d_result);
  free(data);
  free(result);
  return 0;
}
