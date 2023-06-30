#include <algorithm>
#include <chrono>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

typedef unsigned char uchar;

template <typename Td, typename Ts>
__global__
void cvt (      Td *__restrict__ dst,
          const Ts *__restrict__ src,
          const int nelems)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nelems) {
    dst[i] = static_cast<Td>(src[i]);
  }
}

template <typename Td, typename Ts>
void convert(int nelems, int niters)
{
  Ts *src;
  hipMallocManaged((void**)&src, nelems * sizeof(Ts));
  Td *dst;
  hipMallocManaged((void**)&dst, nelems * sizeof(Td));

  const size_t ls = std::min((size_t)nelems, (size_t)256);
  const size_t gs = (nelems + 1) / ls;
  dim3 grid (gs);
  dim3 block (ls);

  // Warm-up run
  hipLaunchKernelGGL(HIP_KERNEL_NAME(cvt<Td, Ts>), grid, block, 0, 0, dst, src, nelems);
  hipDeviceSynchronize();

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < niters; i++) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(cvt<Td, Ts>), grid, block, 0, 0, dst, src, nelems);
  }
  hipDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  double time = std::chrono::duration_cast<std::chrono::microseconds>
                (end - start).count() / niters / 1.0e6;
  double size = (sizeof(Td) + sizeof(Ts)) * nelems / 1e9;
  printf("size(GB):%.2f, average time(sec):%f, BW:%f\n", size, time, size / time);

  hipFree(src);
  hipFree(dst);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int niters = atoi(argv[1]);
  const int nelems = 1024 * 1024 * 256;

  convert<half, hip_bfloat16>(nelems, niters); 
  convert<float, hip_bfloat16>(nelems, niters); 
  convert<int, hip_bfloat16>(nelems, niters); 
  convert<char, hip_bfloat16>(nelems, niters); 
  convert<uchar, hip_bfloat16>(nelems, niters); 

  convert<half, half>(nelems, niters); 
  convert<float, half>(nelems, niters); 
  convert<int, half>(nelems, niters); 
  convert<char, half>(nelems, niters); 
  convert<uchar, half>(nelems, niters); 

  convert<float, float>(nelems, niters); 
  convert<half, float>(nelems, niters); 
  convert<int, float>(nelems, niters); 
  convert<char, float>(nelems, niters); 
  convert<uchar, float>(nelems, niters); 

  convert<int, int>(nelems, niters); 
  convert<float, int>(nelems, niters); 
  convert<half, int>(nelems, niters); 
  convert<char, int>(nelems, niters); 
  convert<uchar, int>(nelems, niters); 

  convert<int, char>(nelems, niters); 
  convert<float, char>(nelems, niters); 
  convert<half, char>(nelems, niters); 
  convert<char, char>(nelems, niters); 
  convert<uchar, char>(nelems, niters); 

  convert<int, uchar>(nelems, niters); 
  convert<float, uchar>(nelems, niters); 
  convert<half, uchar>(nelems, niters); 
  convert<uchar, uchar>(nelems, niters); 
  convert<uchar, uchar>(nelems, niters); 

  return 0;
}
