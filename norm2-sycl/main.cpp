#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <vector>
#include <oneapi/mkl.hpp>
#include "common.h"

long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  // repeat at least once
  const int repeat = max(1, atoi(argv[1]));

  bool ok = true;

#ifdef USE_GPU
  sycl::gpu_selector dev_sel;
#else
  sycl::cpu_selector dev_sel;
#endif
  sycl::queue q(dev_sel);

  // store the nrm2 results
  float* result = (float*) malloc (repeat * sizeof(float));
  if (!result) {
    printf ("result memory allocation failed");
    return 1;
  }

  // store the mkl nrm2 status
  std::vector<sycl::event> status (repeat);

  for (int n = 512*1024; n <= 1024*1024*512; n = n * 2) {
    int i, j;
    size_t size = n * sizeof(float);
    float* a = (float *)malloc (size);
    if (!a) {
      printf ("host memory allocation failed");
      break;
    }

    // reference
    double gold = 0.0;  // double is required to match host and device results 
    for (i = 0; i < n; i++) {
      a[i] = (float)((i+1) % 7);
      gold += a[i]*a[i];
    }
    gold = sqrt(gold);

    long start = get_time();

    float* d_a = (float *)sycl::malloc_device(size, q);

    q.memcpy(d_a, a, size).wait();

    try {
      for (j = 0; j < repeat; j++) {
        status[j] = oneapi::mkl::blas::column_major::nrm2(q, n, d_a, 1, result+j);
      }
    } catch(sycl::exception const& e) {
      std::cout << "\t\tCaught synchronous SYCL exception during NRM2:\n"
                << e.what() << std::endl;
    }

    for (j = 0; j < repeat; j++) status[j].wait();

    sycl::free(d_a, q);

    long end = get_time();
    printf("#elements = %.2f M, measured time = %.3f s\n", 
            n / (1024.f*1024.f), (end-start) / 1e6f);

    if (!a) free(a);

    // snrm2 results match across all iterations
    for (j = 0; j < repeat; j++) 
     if (fabsf((float)gold - result[j]) > 1e-3f) {
       printf("FAIL at iteration %d: gold=%f actual=%f for %d elements\n",
              j, (float)gold, result[j], i);
       ok = false;
       break;
     }
  }

  free(result);

  if (ok) printf("PASS\n");
  return 0;
}
