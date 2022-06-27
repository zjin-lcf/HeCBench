#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "common.h"

void k0 (nd_item<1> &item, const float *__restrict a, float *__restrict o) {
  int t = item.get_global_id(0);
  float x = a[t];
  o[t] = sycl::cosh(x)/sycl::sinh(x) - 1.f/x;
}

void k1 (nd_item<1> &item, const float *__restrict a, float *__restrict o) {
  int t = item.get_global_id(0);
  float x = a[t];
  o[t] = (sycl::exp(x) + sycl::exp(-x)) / (sycl::exp(x) - sycl::exp(-x)) - 1.f/x;
}

void k2 (nd_item<1> &item, const float *__restrict a, float *__restrict o) {
  int t = item.get_global_id(0);
  float x = a[t];
  o[t] = (sycl::exp(2*x) + 1.f) / (sycl::exp(2*x) - 1.f) - 1.f/x;
}

void k3 (nd_item<1> &item, const float *__restrict a, float *__restrict o) {
  int t = item.get_global_id(0);
  float x = a[t];
  o[t] = 1.f / sycl::tanh(x) - 1.f/x;
}

void k4 (nd_item<1> &item, const float *__restrict a, float *__restrict o) {
  int t = item.get_global_id(0);
  float x = a[t];
  float x2 = x * x;
  float x4 = x2 * x2;
  float x6 = x4 * x2;
  o[t] = x * (1.f/3.f - 1.f/45.f * x2 + 2.f/945.f * x4 - 1.f/4725.f * x6);
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

void k5 (nd_item<1> &item, const float *__restrict a, float *__restrict o) {
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

  float *a, *o0, *o1, *o2, *o3, *o4, *o5;

  a = (float*) malloc (size);
  // the range [-1.8, -0.00001)
  for (int i = 0; i < n; i++) {
    a[i] = -1.8f + i * (1.79999f / n);
  }

  o0 = (float*) malloc (size);
  o1 = (float*) malloc (size);
  o2 = (float*) malloc (size);
  o3 = (float*) malloc (size);
  o4 = (float*) malloc (size);
  o5 = (float*) malloc (size);

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  float *d_a, *d_o0, *d_o1, *d_o2, *d_o3, *d_o4, *d_o5;
  d_a = sycl::malloc_device<float>(n, q);
  d_o0 = sycl::malloc_device<float>(n, q);
  d_o1 = sycl::malloc_device<float>(n, q);
  d_o2 = sycl::malloc_device<float>(n, q);
  d_o3 = sycl::malloc_device<float>(n, q);
  d_o4 = sycl::malloc_device<float>(n, q);
  d_o5 = sycl::malloc_device<float>(n, q);

  q.memcpy(d_a, a, size).wait();
  
  range<1> gws (n);
  range<1> lws (256);

  q.wait();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class kernel0>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
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
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class kernel1>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
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
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class kernel2>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        k2(item, d_a, d_o2);
      });
    });
  }
  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of k2: %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class kernel3>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        k3(item, d_a, d_o3);
      });
    });
  }
  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of k3: %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class kernel4>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        k4(item, d_a, d_o4);
      });
    });
  }
  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of k4: %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class kernel5>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        k5(item, d_a, d_o5);
      });
    });
  }
  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of k5: %f (s)\n", (time * 1e-9f) / repeat);

  q.memcpy(o0, d_o0, size).wait();
  q.memcpy(o1, d_o1, size).wait();
  q.memcpy(o2, d_o2, size).wait();
  q.memcpy(o3, d_o3, size).wait();
  q.memcpy(o4, d_o4, size).wait();
  q.memcpy(o5, d_o5, size).wait();
    
  float s01 = 0, s02 = 0, s03 = 0, s04 = 0, s05 = 0;
  float s12 = 0, s13 = 0, s14 = 0, s15 = 0;
  float s23 = 0, s24 = 0, s25 = 0;
  float s34 = 0, s35 = 0;
  float s45 = 0;

  for (int j = 0; j < n; j++) {
    s01 += (o0[j] - o1[j]) * (o0[j] - o1[j]);
    s02 += (o0[j] - o2[j]) * (o0[j] - o2[j]);
    s03 += (o0[j] - o3[j]) * (o0[j] - o3[j]);
    s04 += (o0[j] - o4[j]) * (o0[j] - o4[j]);
    s05 += (o0[j] - o5[j]) * (o0[j] - o5[j]);

    s12 += (o1[j] - o2[j]) * (o1[j] - o2[j]);
    s13 += (o1[j] - o3[j]) * (o1[j] - o3[j]);
    s14 += (o1[j] - o4[j]) * (o1[j] - o4[j]);
    s15 += (o1[j] - o5[j]) * (o1[j] - o5[j]);

    s23 += (o2[j] - o3[j]) * (o2[j] - o3[j]);
    s24 += (o2[j] - o4[j]) * (o2[j] - o4[j]);
    s25 += (o2[j] - o5[j]) * (o2[j] - o5[j]);

    s34 += (o3[j] - o4[j]) * (o3[j] - o4[j]);
    s35 += (o3[j] - o5[j]) * (o3[j] - o5[j]);

    s45 += (o4[j] - o5[j]) * (o4[j] - o5[j]);
  }

  printf("Squared error statistics for six kernels :\n");
  printf("%f (k0-k1) %f (k0-k2) %f (k0-k3) %f (k0-k4) %f (k0-k5)\n",
         s01, s02, s03, s04, s05);
  printf("%f (k1-k2) %f (k1-k3) %f (k1-k4) %f (k1-k5)\n",
         s12, s13, s14, s15);
  printf("%f (k2-k3) %f (k2-k4) %f (k2-k5)\n", s23, s24, s25);
  printf("%f (k3-k4) %f (k3-k5) \n", s34, s35);
  printf("%f (k4-k5) \n", s45);

  free(a);
  free(o0);
  free(o1);
  free(o2);
  free(o3);
  free(o4);
  free(o5);
  sycl::free(d_a, q);
  sycl::free(d_o0, q);
  sycl::free(d_o1, q);
  sycl::free(d_o2, q);
  sycl::free(d_o3, q);
  sycl::free(d_o4, q);
  sycl::free(d_o5, q);
  return 0;
}
