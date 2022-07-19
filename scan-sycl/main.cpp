#include <stdio.h>
#include <chrono>
#include "common.h"

#define N 512

template <typename dataType>
void runTest (const dataType *in, dataType *out, const int n, const int repeat) 
{
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<dataType,1> d_in(in, n);
  buffer<dataType,1> d_out(out, n);

  range<1> lws (n/2);
  range<1> gws (n/2);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      auto g_odata = d_out.template get_access<sycl_discard_write>(cgh);
      auto g_idata = d_in.template get_access<sycl_read>(cgh);
      accessor<dataType, 1, sycl_read_write, access::target::local> temp(N, cgh);
      cgh.parallel_for<class scan_block>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
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
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of block scan: %f (us)\n", (time * 1e-3f) / repeat);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
    
  float in[N];
  float cpu_out[N];
  float gpu_out[N];
  for (int i = 0; i < N; i++) in[i] = (i % 5)+1;

  runTest(in, gpu_out, N, repeat);

  bool ok = true;
  if (gpu_out[0] != 0) {
    ok = false;
  }

  cpu_out[0] = 0;
  for (int i = 1; i < N; i++) 
  {
    cpu_out[i] = cpu_out[i-1] + in[i-1];
    if (cpu_out[i] != gpu_out[i]) { 
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  return 0; 
}
