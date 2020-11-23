#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <CL/sycl.hpp>
#include "common.h"

#define N 512
#define ITERATION 100000

template <typename dataType>
void runTest (const dataType *in, dataType *out, const int n) 
{
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<dataType,1> d_in(in, n);
  buffer<dataType,1> d_out(out, n);

  for (int i = 0; i < ITERATION; i++) {
    q.submit([&] (handler &cgh) {
    auto g_odata = d_out.template get_access<sycl_write>(cgh);
    auto g_idata = d_in.template get_access<sycl_read>(cgh);
    accessor<dataType, 1, sycl_read_write, access::target::local> temp(N, cgh);
    cgh.parallel_for<class scan_block>(nd_range<1>(range<1>(N/2), range<1>(N/2)), [=] (nd_item<1> item) {
      int thid = item.get_local_id(0);
      int offset = 1;
      temp[2*thid]   = g_idata[2*thid];
      temp[2*thid+1] = g_idata[2*thid+1];
      for (int d = n >> 1; d > 0; d >>= 1) 
      {
        item.barrier(access::fence_space::local_space);
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
        item.barrier(access::fence_space::local_space);
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
    });
    });
  }
  q.wait();
}

int main() 
{
  float in[N];
  float cpu_out[N];
  float gpu_out[N];
  int error = 0;
  for (int i = 0; i < N; i++) in[i] = (i % 5)+1;
  runTest(in, gpu_out, N); 
  cpu_out[0] = 0;
  if (gpu_out[0] != 0) {
   error++;
   printf("gpu = %f at index 0\n", gpu_out[0]);
  }
  for (int i = 1; i < N; i++) 
  {
    cpu_out[i] = cpu_out[i-1] + in[i-1];
    if (cpu_out[i] != gpu_out[i]) { 
     error++;
     printf("cpu = %f gpu = %f at index %d\n",
     cpu_out[i], gpu_out[i], i);
    }
  }
  if (error == 0) printf("PASS\n");
  return 0; 
}


