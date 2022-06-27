#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include "common.h"
#include "reference.h"

int main(int argc, char* argv[]) {

  if (argc != 4) {
    printf("Usage: ./%s <query length> <subject length> <repeat>\n", argv[0]);
    return -1;
  }

  const int M = atoi(argv[1]);
  const int N = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  printf("Query length = %d\n", M);
  printf("Subject length = %d\n", N);

  // host side memory
  float *subject = (float*) malloc (sizeof(float)*N);
  float *lower = (float*) malloc (sizeof(float)*N);
  float *upper = (float*) malloc (sizeof(float)*N);
  float *lb = (float*) malloc (sizeof(float)*(N-M+1));
  float *lb_h = (float*) malloc (sizeof(float)*(N-M+1));
  float *avgs = (float*) malloc (sizeof(float)*(N-M+1));
  float *stds = (float*) malloc (sizeof(float)*(N-M+1));

  srand(123);
  for (int i = 0; i < N; ++i) subject[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < N-M+1; ++i) avgs[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < N-M+1; ++i) stds[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < M; ++i) upper[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < M; ++i) lower[i] = (float)rand() / (float)RAND_MAX;

  { // sycl scope
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<const float, 1> d_subject (subject, N);
  buffer<const float, 1> d_avgs (avgs, N-M+1);
  buffer<const float, 1> d_stds (stds, N-M+1);
  buffer<      float, 1> d_lb (lb, N-M+1);
  buffer<const float, 1> d_lower (lower, N);
  buffer<const float, 1> d_upper (upper, N);

  const int blocks = 256;
  const int grids = (N-M+1 + blocks - 1) / blocks;
  range<1> gws (grids * blocks);
  range<1> lws (blocks);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      auto subject = d_subject.get_access<sycl_read>(cgh);
      auto avgs = d_avgs.get_access<sycl_read>(cgh);
      auto stds = d_stds.get_access<sycl_read>(cgh);
      auto lb = d_lb.get_access<sycl_discard_write>(cgh);
      auto lower_bound = d_lower.get_access<sycl_read>(cgh);
      auto upper_bound = d_upper.get_access<sycl_read>(cgh);
      accessor<float, 1, sycl_read_write, access::target::local> cache(M+blocks, cgh);
      cgh.parallel_for<class lp_koegh>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        int lid = item.get_local_id(0);
        int blockDim = item.get_local_range(0);
        int blockIdx = item.get_group(0);
        int blockSize = blockDim * blockIdx;
        int idx = blockSize + lid;

        for (int k = lid; k < blockDim + M; k += blockDim)
          if (blockSize + k < N) {
            cache[k] = subject[blockSize + k];
          }

        item.barrier(access::fence_space::local_space);

        if (idx < N-M+1) {

          // obtain statistics
          float residues = 0;
          float avg = avgs[idx];
          float std = stds[idx];

          for (int i = 0; i < M; ++i) {
            // differences to envelopes
            float value = (cache[lid+i] - avg) / std;
            float lower = value - lower_bound[i];
            float upper = value - upper_bound[i];

            // Euclidean distance
            residues += upper*upper*(upper > 0) + lower*lower*(lower < 0);
          }

          lb[idx] = residues;
        }
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  }

  // verify
  reference(subject, avgs, stds, lb_h, lower, upper, M, N);
  bool ok = true;
  for (int i = 0; i < N-M+1; i++) {
    if (fabs(lb[i] - lb_h[i]) > 1e-3) {
      printf("%d %f %f\n", i, lb[i], lb_h[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(lb);
  free(lb_h);
  free(avgs);
  free(stds);
  free(subject);
  free(lower);
  free(upper);
  return 0;
}


