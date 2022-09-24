#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cuda.h>

// scan over N elements
#define N 512

template<typename T>
void verify(const T* cpu_out, const T* gpu_out, int n)
{
  int error = memcmp(cpu_out, gpu_out, n * sizeof(T));
  printf("%s\n", error ? "FAIL" : "PASS");
}

// bank conflict aware optimization

#define LOG_MEM_BANKS 5
#define OFFSET(n) ((n) >> LOG_MEM_BANKS)

template<typename T>
__global__ void prescan_bcao (
        T *__restrict__ g_odata,
  const T *__restrict__ g_idata,
  const int n)
{
  __shared__ T temp[2*N];
  int thid = threadIdx.x; 
  int a = thid;
  int b = a + (n/2);
  int oa = OFFSET(a);
  int ob = OFFSET(b);

  temp[a + oa] = g_idata[a];
  temp[b + ob] = g_idata[b];

  int offset = 1;
  for (int d = n >> 1; d > 0; d >>= 1) 
  {
    __syncthreads();
    if (thid < d) 
    {
      int ai = offset*(2*thid+1)-1;
      int bi = offset*(2*thid+2)-1;
      ai += OFFSET(ai);
      bi += OFFSET(bi);
      temp[bi] += temp[ai];
    }
    offset *= 2;
  }

  if (thid == 0) temp[n-1+OFFSET(n-1)] = 0; // clear the last elem
  for (int d = 1; d < n; d *= 2) // traverse down
  {
    offset >>= 1;     
    __syncthreads();      
    if (thid < d)
    {
      int ai = offset*(2*thid+1)-1;
      int bi = offset*(2*thid+2)-1;
      ai += OFFSET(ai);
      bi += OFFSET(bi);
      T t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads(); // required

  g_odata[a] = temp[a + oa];
  g_odata[b] = temp[b + ob];
}

template<typename T>
__global__ void prescan(
        T *__restrict__ g_odata,
  const T *__restrict__ g_idata,
  const int n)
{
  __shared__ T temp[N];
  int thid = threadIdx.x; 
  int offset = 1;
  temp[2*thid]   = g_idata[2*thid];
  temp[2*thid+1] = g_idata[2*thid+1];
  for (int d = n >> 1; d > 0; d >>= 1) 
  {
    __syncthreads();
    if (thid < d) 
    {
      int ai = offset*(2*thid+1)-1;
      int bi = offset*(2*thid+2)-1;
      temp[bi] += temp[ai];
    }
    offset *= 2;
  }

  if (thid == 0) temp[n-1] = 0; // clear the last elem
  for (int d = 1; d < n; d *= 2) // traverse down
  {
    offset >>= 1;     
    __syncthreads();      
    if (thid < d)
    {
      int ai = offset*(2*thid+1)-1;
      int bi = offset*(2*thid+2)-1;
      float t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  g_odata[2*thid] = temp[2*thid];
  g_odata[2*thid+1] = temp[2*thid+1];
}

template <typename T>
void runTest (const int repeat, bool timing = false) 
{
  T in[N];
  T cpu_out[N];
  T gpu_out[N];

  int n = N;

  for (int i = 0; i < n; i++) in[i] = (i % 5)+1;
  cpu_out[0] = 0;
  for (int i = 1; i < n; i++) cpu_out[i] = cpu_out[i-1] + in[i-1];

  T *d_in, *d_out;

  cudaMalloc((void**)&d_in, n*sizeof(T));
  cudaMemcpy(d_in, in, n*sizeof(T), cudaMemcpyHostToDevice); 

  cudaMalloc((void**)&d_out, n*sizeof(T));

  dim3 grids (1);
  dim3 blocks (n/2);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    prescan<<<grids, blocks>>>(d_out, d_in, n);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  if (timing) {
    printf("Element size in bytes is %zu. Average execution time of block scan (w/  bank conflicts): %f (us)\n",
           sizeof(T), (time * 1e-3f) / repeat);
  }
  cudaMemcpy(gpu_out, d_out, n*sizeof(T), cudaMemcpyDeviceToHost);
  if (!timing) verify(cpu_out, gpu_out, n);

  // bcao
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    prescan_bcao<<<grids, blocks>>>(d_out, d_in, n);
  }

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  auto bcao_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  if (timing) {
    printf("Element size in bytes is %zu. Average execution time of block scan (w/o bank conflicts): %f (us). ",
           sizeof(T), (bcao_time * 1e-3f) / repeat);
    printf("Reduce the time by %.1f%%\n", (time - bcao_time) * 1.0 / time * 100);
  }
  cudaMemcpy(gpu_out, d_out, n*sizeof(T), cudaMemcpyDeviceToHost);
  if (!timing) verify(cpu_out, gpu_out, n);

  cudaFree(d_in);
  cudaFree(d_out);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
    
  for (int i = 0; i < 2; i++) {
    bool timing = i > 0;
    runTest<char>(repeat, timing);
    runTest<short>(repeat, timing);
    runTest<int>(repeat, timing);
    runTest<long>(repeat, timing);
  }

  return 0; 
}
