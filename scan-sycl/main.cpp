#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include "common.h"

// scan over N elements
#define N 512

template<typename T>
void verify(const T* cpu_out, const T* gpu_out, int n)
{
  int error = memcmp(cpu_out, gpu_out, n * sizeof(T));
  printf("%s\n", error ? "FAIL" : "PASS");
}

#define LOG_MEM_BANKS 5
#define OFFSET(n) ((n) >> LOG_MEM_BANKS)
#define __syncthreads() item.barrier(access::fence_space::local_space)

template<typename T>
class opt_block_scan;

template <typename T>
class block_scan;

// bank conflict aware optimization
template<typename T>
__attribute__((always_inline))
void prescan_bcao (
        nd_item<1> &item,
        local_ptr<T> temp,
        T *__restrict g_odata,
  const T *__restrict g_idata,
  const int n)
{
  int thid = item.get_local_id(0);
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
__attribute__((always_inline))
void prescan (
        nd_item<1> &item,
        local_ptr<T> temp,
        T *__restrict g_odata,
  const T *__restrict g_idata,
  const int n)
{
  int thid = item.get_local_id(0);
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
void runTest (queue &q, const int repeat, bool timing = false) 
{
  T in[N];
  T cpu_out[N];
  T gpu_out[N];

  int n = N;

  for (int i = 0; i < n; i++) in[i] = (i % 5)+1;
  cpu_out[0] = 0;
  for (int i = 1; i < n; i++) cpu_out[i] = cpu_out[i-1] + in[i-1];

  T *d_in = malloc_device<T>(n, q);
  q.memcpy(d_in, in, n*sizeof(T));

  T *d_out = malloc_device<T>(n, q);

  range<1> lws (n/2);
  range<1> gws (n/2);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      accessor<T, 1, sycl_read_write, access::target::local> temp (N, cgh);
      cgh.parallel_for<class block_scan<T>>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        prescan(item, temp.get_pointer(), d_out, d_in, n);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  if (timing) {
    printf("Element size in bytes is %zu. Average execution time of block scan (w/  bank conflicts): %f (us)\n",
           sizeof(T), (time * 1e-3f) / repeat);
  }
  q.memcpy(gpu_out, d_out, n*sizeof(T)).wait();
  if (!timing) verify(cpu_out, gpu_out, n);

  // bcao
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      accessor<T, 1, sycl_read_write, access::target::local> temp (N*2, cgh);
      cgh.parallel_for<class opt_block_scan<T>>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        prescan_bcao(item, temp.get_pointer(), d_out, d_in, n);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  auto bcao_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  if (timing) {
    printf("Element size in bytes is %zu. Average execution time of block scan (w/o bank conflicts): %f (us). ",
           sizeof(T), (bcao_time * 1e-3f) / repeat);
    printf("Reduce the time by %.1f%%\n", (time - bcao_time) * 1.0 / time * 100);
  }
  q.memcpy(gpu_out, d_out, n*sizeof(T)).wait();
  if (!timing) verify(cpu_out, gpu_out, n);

  free(d_in, q);
  free(d_out, q);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);
    
  for (int i = 0; i < 2; i++) {
    bool timing = i > 0;
    runTest<char>(q, repeat, timing);
    runTest<short>(q, repeat, timing);
    runTest<int>(q, repeat, timing);
    runTest<long>(q, repeat, timing);
  }

  return 0; 
}
