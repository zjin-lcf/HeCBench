#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <sycl/sycl.hpp>

void k0 (sycl::nd_item<1> &item, const float *__restrict a, float *__restrict o) {
  int t = item.get_global_id(0);
  float x = a[t];
  o[t] = sycl::cosh(x)/sycl::sinh(x) - 1.f/x;
}

void k1 (sycl::nd_item<1> &item, const float *__restrict a, float *__restrict o) {
  int t = item.get_global_id(0);
  float x = a[t];
  o[t] = 1.f / sycl::tanh(x) - 1.f/x;
}

/*
Copyright (c) 2018-2021, Norbert Juffa
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

  1. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

void k2 (sycl::nd_item<1> &item, const float *__restrict a, float *__restrict o) {
  int t = item.get_global_id(0);
  float x = a[t];
  float s, r;
  s = x * x;
  r =                   7.70960469e-8f;
  r = sycl::fma (r, s, -1.65101926e-6f);
  r = sycl::fma (r, s,  2.03457112e-5f);
  r = sycl::fma (r, s, -2.10521728e-4f);
  r = sycl::fma (r, s,  2.11580913e-3f);
  r = sycl::fma (r, s, -2.22220998e-2f);
  r = sycl::fma (r, s,  8.33333284e-2f);
  r = sycl::fma (r, x,  0.25f * x);
  o[t] = r;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage %s <n> <repeat>\n", argv[0]);
    return 1;
  }

  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  const size_t size = sizeof(float) * n;

  float *a, *o, *o0, *o1, *o2;

  a = (float*) malloc (size);
  // the range [-1.8, -0.00001)
  for (int i = 0; i < n; i++) {
    a[i] = -1.8f + i * (1.79999f / n);
  }

  o = (float*) malloc (size);
  o0 = (float*) malloc (size);
  o1 = (float*) malloc (size);
  o2 = (float*) malloc (size);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_a, *d_o0, *d_o1, *d_o2;
  d_a = sycl::malloc_device<float>(n, q);
  d_o0 = sycl::malloc_device<float>(n, q);
  d_o1 = sycl::malloc_device<float>(n, q);
  d_o2 = sycl::malloc_device<float>(n, q);

  q.memcpy(d_a, a, size).wait();

  sycl::range<1> gws (n);
  sycl::range<1> lws (256);

  q.wait();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class kernel0>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        k0(item, d_a, d_o0);
      });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of k0: %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class kernel1>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        k1(item, d_a, d_o1);
      });
    });
  }
  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of k1: %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class kernel2>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        k2(item, d_a, d_o2);
      });
    });
  }
  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of k2: %f (s)\n", (time * 1e-9f) / repeat);

  q.memcpy(o0, d_o0, size).wait();
  q.memcpy(o1, d_o1, size).wait();
  q.memcpy(o2, d_o2, size).wait();

  // https://en.wikipedia.org/wiki/Brillouin_and_Langevin_functions
  for (int i = 0; i < n; i++) {
    float x = a[i];
    float x2 = x * x;
    float x4 = x2 * x2;
    float x6 = x4 * x2;
    o[i] = x * (1.f/3.f - 1.f/45.f * x2 + 2.f/945.f * x4 - 1.f/4725.f * x6);
  }

  float e[3] = {0,0,0};

  for (int i = 0; i < n; i++) {
    e[0] += (o[i] - o0[i]) * (o[i] - o0[i]);
    e[1] += (o[i] - o1[i]) * (o[i] - o1[i]);
    e[2] += (o[i] - o2[i]) * (o[i] - o2[i]);
  }

  printf("\nError statistics for the kernels:\n");
  for (int i = 0; i < 3; i++) {
    printf("%f ", sqrt(e[i]));
  }
  printf("\n");

  free(a);
  free(o);
  free(o0);
  free(o1);
  free(o2);
  sycl::free(d_a, q);
  sycl::free(d_o0, q);
  sycl::free(d_o1, q);
  sycl::free(d_o2, q);
  return 0;
}
