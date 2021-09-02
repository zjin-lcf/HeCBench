/* Reference: https://x.momo86.net/en?p=113 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define NUM_OF_BLOCKS 1024
#define NUM_OF_THREADS 256

__device__ half2 half_max(const half2 a, const half2 b) {
  const half2 sub = __hsub2(a, b);
  const unsigned sign = (*reinterpret_cast<const unsigned*>(&sub)) & 0x80008000u;
  const unsigned sw = 0x00003210 | (((sign >> 21) | (sign >> 13)) * 0x11);
  const unsigned int res = __byte_perm(*reinterpret_cast<const unsigned*>(&a), 
      *reinterpret_cast<const unsigned*>(&b), sw);
  return *reinterpret_cast<const half2*>(&res);
}

__device__ half half_max(const half a, const half b) {
  const half sub = __hsub(a, b);
  const unsigned sign = (*reinterpret_cast<const short*>(&sub)) & 0x8000u;
  const unsigned sw = 0x00000010 | ((sign >> 13) * 0x11);
  const unsigned short res = __byte_perm(*reinterpret_cast<const short*>(&a), 
      *reinterpret_cast<const short*>(&b), sw);
  return *reinterpret_cast<const half*>(&res);
}

template <typename T>
__global__
void hmax(T const *__restrict__ const a,
          T const *__restrict__ const b,
          T *__restrict__ const r,
          const size_t size)
{
  for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; 
              i < size; i += blockDim.x * gridDim.x)
    r[i] = half_max(a[i], b[i]);
}


void generateInput(half2 * a, size_t size)
{
  for (size_t i = 0; i < size; ++i)
  {
    half2 temp;
    temp.x = static_cast<float>(rand() % 922021);
    temp.y = static_cast<float>(rand() % 922021);
    a[i] = temp;
  }
}

// compute the maximum of two values
int main(int argc, char *argv[])
{
  size_t size = NUM_OF_BLOCKS*NUM_OF_THREADS*16;

  half2 * a, *b, *r;
  half2 * d_a, *d_b, *d_r;

  a = (half2*) malloc (size*sizeof(half2));
  b = (half2*) malloc (size*sizeof(half2));
  r = (half2*) malloc (size*sizeof(half2));

  hipMalloc((void**)&d_a, size*sizeof(half2));
  hipMalloc((void**)&d_b, size*sizeof(half2));
  hipMalloc((void**)&d_r, size*sizeof(half2));

  // initialize input values
  srand(123); 
  generateInput(a, size);
  hipMemcpy(d_a, a, size*sizeof(half2), hipMemcpyHostToDevice);

  generateInput(b, size);
  hipMemcpy(d_b, b, size*sizeof(half2), hipMemcpyHostToDevice);


  // run hmax2
  for (int i = 0; i < 100; i++)
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hmax<half2>), dim3(NUM_OF_BLOCKS), dim3(NUM_OF_THREADS), 0, 0, 
      d_a, d_b, d_r, size);

  // verify
  hipMemcpy(r, d_r, size*sizeof(half2), hipMemcpyDeviceToHost);

  bool ok = true;
  for (size_t i = 0; i < size; ++i)
  {
    float2 fa = __half22float2(a[i]);
    float2 fb = __half22float2(b[i]);
    float2 fr = __half22float2(r[i]);
    float x = fmaxf(fa.x, fb.x);
    float y = fmaxf(fa.y, fb.y);
    if (fabsf(fr.x - x) > 1e-3 || fabsf(fr.y - y) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("fp16_hmax2 %s\n", ok ?  "PASSED" : "FAILED");


  // run hmax (the size is doubled)
  for (int i = 0; i < 100; i++)
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hmax<half>), dim3(NUM_OF_BLOCKS), dim3(NUM_OF_THREADS), 0, 0, 
      (half*)d_a, (half*)d_b, (half*)d_r, size*2);

  // verify
  ok = true;
  for (size_t i = 0; i < size; ++i)
  {
    float2 fa = __half22float2(a[i]);
    float2 fb = __half22float2(b[i]);
    float2 fr = __half22float2(r[i]);
    float x = fmaxf(fa.x, fb.x);
    float y = fmaxf(fa.y, fb.y);
    if (fabsf(fr.x - x) > 1e-3 || fabsf(fr.y - y) > 1e-3) {
      ok = false;
      break;
    }
  }

  printf("fp16_hmax %s\n", ok ?  "PASSED" : "FAILED");

  hipFree(d_a);
  hipFree(d_b);
  hipFree(d_r);
  free(a);
  free(b);
  free(r);

  return EXIT_SUCCESS;
}
