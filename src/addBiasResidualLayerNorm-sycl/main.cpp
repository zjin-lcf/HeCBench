/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <chrono>
#include <random>
#include <sycl/sycl.hpp>
#include "kernels.h"

template<typename T, int V>
void invokeAddBiasResidualLayerNorm(
    sycl::queue &q,
          T*     out,
    const T*     input,
    const T*     bias,
    const T*     gamma,
    const T*     beta,
    const float  layernorm_eps,
    int          m,
    int          n)
{
  if (V == 2) {

    sycl::range<1> gws (m * n / 8);
    sycl::range<1> lws (n / 8);

    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<float, 1> shared(sycl::range<1>(32), cgh);
      sycl::local_accessor<float, 0> s_mean(cgh);
      sycl::local_accessor<float, 0> s_variance(cgh);
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item)
        [[intel::reqd_sub_group_size(32)]] {
        addBiasResidualPostLayerNormV2<T>(
          out, input, bias, gamma, beta, layernorm_eps,
          n, item, shared.get_pointer(),
          s_mean, s_variance);
      });
    });
  }
  else {
    sycl::range<1> gws (m * std::min(n, 256));
    sycl::range<1> lws (std::min(n, 256));

    int num_trips = (n + lws[0] - 1) / lws[0];
    if (num_trips == 1) {
      q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared(sycl::range<1>(32), cgh);
        sycl::local_accessor<float, 0> s_mean(cgh);
        sycl::local_accessor<float, 0> s_variance(cgh);

        cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item)
          [[intel::reqd_sub_group_size(32)]] {
          addBiasResidualPostLayerNorm<T, 1>(
            out, input, bias, gamma, beta, layernorm_eps,
            n, item, shared.get_pointer(),
            s_mean, s_variance);
        });
      });
    }
    else if (num_trips == 2) {
      q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared(sycl::range<1>(32), cgh);
        sycl::local_accessor<float, 0> s_mean(cgh);
        sycl::local_accessor<float, 0> s_variance(cgh);

        cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item)
          [[intel::reqd_sub_group_size(32)]] {
          addBiasResidualPostLayerNorm<T, 2>(
            out, input, bias, gamma, beta, layernorm_eps,
            n, item, shared.get_pointer(),
            s_mean, s_variance);
        });
      });
    }
    else {
      q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared(sycl::range<1>(32), cgh);
        sycl::local_accessor<float, 0> s_mean(cgh);
        sycl::local_accessor<float, 0> s_variance(cgh);

        cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item)
          [[intel::reqd_sub_group_size(32)]] {
          generalAddBiasResidualPostLayerNorm<T>(
            out, input, bias, gamma, beta, layernorm_eps,
            n, item, shared.get_pointer(),
            s_mean, s_variance);
        });
      });
    }
  }
}

template<typename T, int V>
void layer(int repeat) {
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  const int m = 4096;  // batch size

  int dim[] = {256, 512, 768, 1024, 2048, 4096, 8192};

  for (int i = 0; i < sizeof(dim) / sizeof(int); i++) {

    std::mt19937 gen (19937);
    std::uniform_real_distribution<float> dis(0.f, 1.f);

    // n-dimensional data
    const int n = dim[i]; 
    const int input_size = m * n;
    const int output_size = m * n;
    const int input_size_bytes = input_size * sizeof(T);
    const int output_size_bytes = output_size * sizeof(T);
    const int bias_size_bytes = n * sizeof(T);
    const int beta_size_bytes = n * sizeof(T);
    const int gamma_size_bytes = n * sizeof(T);

    T *h_input = (T*) malloc (input_size_bytes);
    T *h_output = (T*) malloc (output_size_bytes);
    T *h_bias = (T*) malloc (bias_size_bytes);
    T *h_gamma = (T*) malloc (gamma_size_bytes);
    T *h_beta = (T*) malloc (beta_size_bytes);

    for (int i = 0; i < input_size; i++) {
      h_input[i] = (T) dis(gen);
    }
    for (int i = 0; i < n; i++) {
      h_bias[i] = (T) dis(gen);
      h_gamma[i] = (T) dis(gen);
      h_beta[i] = (T) dis(gen);
    }

    float layernorm_eps = 1e-6;
    T *d_input, *d_output, *d_bias, *d_gamma, *d_beta;
    d_input = (T *)sycl::malloc_device(input_size_bytes, q);
    q.memcpy(d_input, h_input, input_size_bytes);

    d_output = (T *)sycl::malloc_device(output_size_bytes, q);
    q.memset(d_output, 0, output_size_bytes);

    d_bias = (T *)sycl::malloc_device(bias_size_bytes, q);
    q.memcpy(d_bias, h_bias, bias_size_bytes);

    d_gamma = (T *)sycl::malloc_device(gamma_size_bytes, q);
    q.memcpy(d_gamma, h_gamma, gamma_size_bytes);

    d_beta = (T *)sycl::malloc_device(beta_size_bytes, q);
    q.memcpy(d_beta, h_beta, beta_size_bytes);

    q.wait();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      invokeAddBiasResidualLayerNorm<T, V>(
        q, d_output, d_input, d_bias, d_gamma, d_beta, layernorm_eps, m, n);
    }
    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of AddBiasResidualLayerNorm (%d x %d): %f (us)\n",
           m, n, (time * 1e-3f) / repeat);

    q.memcpy(h_output, d_output, output_size_bytes).wait();

    float s = 0;
    for (int i = 0; i < output_size; i++)
      s += float(h_output[i]);

    printf("Checksum = %f\n", s / (n * n));

    sycl::free(d_input, q);
    sycl::free(d_output, q);
    sycl::free(d_bias, q);
    sycl::free(d_gamma, q);
    sycl::free(d_beta, q);

    free(h_input);
    free(h_output);
    free(h_bias);
    free(h_gamma);
    free(h_beta);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  printf("---------------- float16 (version 1) -------------\n");
  layer<sycl::half, 1>(repeat);
  printf("---------------- float16 (version 2) -------------\n");
  layer<sycl::half, 2>(repeat);

  printf("---------------- bfloat16 (version 1) -------------\n");
  layer<sycl::ext::oneapi::bfloat16, 1>(repeat);
  printf("---------------- bfloat16 (version 2) -------------\n");
  layer<sycl::ext::oneapi::bfloat16, 2>(repeat);

  return 0;
}
