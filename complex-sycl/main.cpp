#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "common.h"
#include "complex.h"
#include "kernels.h"

bool check (const char *cs, int n)
{
  bool ok = true;
  for (int i = 0; i < n; i++) {
    if (cs[i] != 5) {
      ok = false; 
      break;
    }
  }
  return ok;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <size> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  char* cs = (char*) malloc (n);

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  char* d_cs = sycl::malloc_device<char>(n, q);

  range<1> gws ((n + 255)/256*256); 
  range<1> lws (256);

  // warmup
  q.submit([&] (handler &cgh) {
    cgh.parallel_for(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      complex_float(item, d_cs, n);
    });
  });

  q.submit([&] (handler &cgh) {
    cgh.parallel_for(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      complex_double(item, d_cs, n);
    });
  });

  q.wait();

  auto start = std::chrono::steady_clock::now();

  // complex numbers in single precision
  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      cgh.parallel_for(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        complex_float(item, d_cs, n);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (float) %f (s)\n", time * 1e-9f / repeat);

  q.memcpy(cs, d_cs, n).wait();
  bool complex_float_check = check(cs, n);

  start = std::chrono::steady_clock::now();

  // complex numbers in double precision
  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      cgh.parallel_for(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        complex_double(item, d_cs, n);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (double) %f (s)\n", time * 1e-9f / repeat);

  q.memcpy(cs, d_cs, n).wait();
  bool complex_double_check = check(cs, n);

  printf("%s\n", (complex_float_check && complex_double_check)
                 ? "PASS" : "FAIL");

  sycl::free(d_cs, q);
  free(cs);

  return 0;
}
