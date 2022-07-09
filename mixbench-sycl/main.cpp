/**
 * main-omp.cpp: This file is the modified read-only mixbench GPU micro-benchmark suite.
 *
 **/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "common.h"

#define VECTOR_SIZE (8*1024*1024)
#define granularity (8)
#define fusion_degree (4)
#define seed 0.1f

void benchmark_func(nd_item<1> &item,
                    float *g_data, const int block_dim,
                    const int compute_iterations)
{
  const unsigned int blockSize = block_dim;
  const int stride = blockSize;
  int idx = item.get_group(0)*blockSize*granularity + item.get_local_id(0);
  const int big_stride = item.get_group_range(0)*blockSize*granularity;

  float tmps[granularity];
  for(int k=0; k<fusion_degree; k++){
    #pragma unroll
    for(int j=0; j<granularity; j++){
      // Load elements (memory intensive part)
      tmps[j] = g_data[idx+j*stride+k*big_stride];
      // Perform computations (compute intensive part)
      for(int i=0; i<compute_iterations; i++)
        tmps[j] = tmps[j]*tmps[j]+(float)seed;
    }
    // Multiply add reduction
    float sum = 0;
    #pragma unroll
    for(int j=0; j<granularity; j+=2)
      sum += tmps[j]*tmps[j+1];
    #pragma unroll
    for(int j=0; j<granularity; j++)
      g_data[idx+k*big_stride] = sum;
  }
}

void mixbenchGPU(long size, int repeat) {
  const char *benchtype = "compute with global memory (block strided)";
  printf("Trade-off type:%s\n", benchtype);
  float *cd = (float*) malloc (size*sizeof(float));
  for (int i = 0; i < size; i++) cd[i] = 0;

  const long reduced_grid_size = size/granularity/128;
  const int block_dim = 256;
  const int grid_dim = reduced_grid_size;

  {
#ifdef USE_GPU 
    gpu_selector dev_sel;
#else
    cpu_selector dev_sel;
#endif
    queue q(dev_sel);

    buffer<float, 1> d_cd (cd, size);

    range<1> gws (grid_dim);
    range<1> lws (block_dim);

    for (int i = 0; i < repeat; i++) {
      q.submit([&](handler &h) {
        auto g_data = d_cd.get_access<sycl_read_write>(h);
        h.parallel_for<class mixbench_warmup>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
          benchmark_func(item, g_data.get_pointer(), block_dim, i);
        });
      });
    }
    q.wait();

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      q.submit([&](handler &h) {
        auto g_data = d_cd.get_access<sycl_read_write>(h);
        h.parallel_for<class mixbench_timing>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
          benchmark_func(item, g_data.get_pointer(), block_dim, i);
        });
      });
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Total kernel execution time: %f (s)\n", time * 1e-9f);
  }

  // verification
  bool ok = true;
  for (int i = 0; i < size; i++) {
    if (cd[i] != 0) {
      if (fabsf(cd[i] - 0.050807f) > 1e-6f) {
        ok = false;
        printf("Verification failed at index %d: %f\n", i, cd[i]);
        break;
      }
    }
  }

  free(cd);
  printf("%s\n", ok ? "PASS" : "FAIL");
}


int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  unsigned int datasize = VECTOR_SIZE*sizeof(float);

  printf("Buffer size: %dMB\n", datasize/(1024*1024));

  mixbenchGPU(VECTOR_SIZE, repeat);

  return 0;
}
