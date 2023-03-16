#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include "common.h"

template<typename T>
void verify(const T* cpu_out, const T* gpu_out, size_t n)
{
  int error = memcmp(cpu_out, gpu_out, n * sizeof(T));
  printf("%s\n", error ? "FAIL" : "PASS");
}

#define LOG_MEM_BANKS 5
#define OFFSET(n) ((n) >> LOG_MEM_BANKS)
#define __syncthreads() item.barrier(access::fence_space::local_space)

template<typename T, int N>
class opt;

template <typename T, int N>
class base;

// bank conflict aware optimization
template<typename T, int N>
void scan_bcao (
        nd_item<1> &item,
        local_ptr<T> temp,
        T *__restrict g_odata,
  const T *__restrict g_idata)
{
  int bid = item.get_group(0);
  g_idata += bid * N;
  g_odata += bid * N;

  int thid = item.get_local_id(0);
  int a = thid;
  int b = a + (N/2);
  int oa = OFFSET(a);
  int ob = OFFSET(b);

  temp[a + oa] = g_idata[a];
  temp[b + ob] = g_idata[b];

  int offset = 1;
  for (int d = N >> 1; d > 0; d >>= 1) 
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

  if (thid == 0) temp[N-1+OFFSET(N-1)] = 0; // clear the last elem
  for (int d = 1; d < N; d *= 2) // traverse down
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

template<typename T, int N>
void scan (
        nd_item<1> &item,
        local_ptr<T> temp,
        T *__restrict g_odata,
  const T *__restrict g_idata)
{
  int bid = item.get_group(0);
  g_idata += bid * N;
  g_odata += bid * N;

  int thid = item.get_local_id(0);
  int offset = 1;
  temp[2*thid]   = g_idata[2*thid];
  temp[2*thid+1] = g_idata[2*thid+1];
  for (int d = N >> 1; d > 0; d >>= 1) 
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

  if (thid == 0) temp[N-1] = 0; // clear the last elem
  for (int d = 1; d < N; d *= 2) // traverse down
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


template <typename T, int N>
void runTest (queue &q, const size_t n, const int repeat, bool timing = false) 
{
  const size_t num_blocks = (n + N - 1) / N;

  const size_t nelems = num_blocks * N; // actual total number of elements

  size_t bytes = nelems * sizeof(T);

  T *in = (T*) malloc (bytes);
  T *cpu_out = (T*) malloc (bytes);
  T *gpu_out = (T*) malloc (bytes);

  srand(123);
  for (size_t n = 0; n < nelems; n++) in[n] = rand() % 5 + 1;

  T *t_in = in;
  T *t_out = cpu_out;
  for (size_t n = 0; n < num_blocks; n++) { 
    t_out[0] = 0;
    for (int i = 1; i < N; i++) 
      t_out[i] = t_out[i-1] + t_in[i-1];
    t_out += N;
    t_in += N;
  }

  T *d_in = malloc_device<T>(nelems, q);
  q.memcpy(d_in, in, bytes);

  T *d_out = malloc_device<T>(nelems, q);

  range<1> gws (num_blocks * N/2);
  range<1> lws (N/2);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      accessor<T, 1, sycl_read_write, access::target::local> temp (N, cgh);
      cgh.parallel_for<class base<T, N>>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        scan<T, N>(item, temp.get_pointer(), d_out, d_in);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  if (timing) {
    printf("Element size in bytes is %zu. Average execution time of scan (w/  bank conflicts): %f (us)\n",
           sizeof(T), (time * 1e-3f) / repeat);
  }
  q.memcpy(gpu_out, d_out, bytes).wait();
  if (!timing) verify(cpu_out, gpu_out, nelems);

  // bcao
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      accessor<T, 1, sycl_read_write, access::target::local> temp (N*2, cgh);
      cgh.parallel_for<class opt<T, N>>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        scan_bcao<T, N>(item, temp.get_pointer(), d_out, d_in);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  auto bcao_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  if (timing) {
    printf("Element size in bytes is %zu. Average execution time of scan (w/o bank conflicts): %f (us). ",
           sizeof(T), (bcao_time * 1e-3f) / repeat);
    printf("Reduce the time by %.1f%%\n", (time - bcao_time) * 1.0 / time * 100);
  }
  q.memcpy(gpu_out, d_out, bytes).wait();
  if (!timing) verify(cpu_out, gpu_out, nelems);

  free(d_in, q);
  free(d_out, q);
  free(in);
  free(cpu_out);
  free(gpu_out);
}

template<int N>
void run (queue &q, const int n, const int repeat) {
  for (int i = 0; i < 2; i++) {
    bool report_timing = i > 0;
    printf("\nThe number of elements to scan in a thread block: %d\n", N);
    runTest< char, N>(q, n, repeat, report_timing);
    runTest<short, N>(q, n, repeat, report_timing);
    runTest<  int, N>(q, n, repeat, report_timing);
    runTest< long, N>(q, n, repeat, report_timing);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);
    
  run< 128>(q, n, repeat);  
  run< 256>(q, n, repeat);  
  run< 512>(q, n, repeat);  
  run<1024>(q, n, repeat);  
  run<2048>(q, n, repeat);  

  return 0; 
}
